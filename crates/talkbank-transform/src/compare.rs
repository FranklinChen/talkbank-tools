//! Transcript comparison: main vs gold-standard reference.
//!
//! Extracts words from both transcripts, normalizes them via
//! [`wer_conform::conform_words`], runs DP alignment, and produces
//! per-utterance comparison annotations with accuracy metrics.
//!
//! This is the Rust implementation of Python's `CompareEngine` +
//! `CompareAnalysisEngine` from batchalign2.

use std::collections::{BTreeMap, HashMap};

use talkbank_model::Span;
use talkbank_model::UtteranceIdx;
use talkbank_model::WriteChat;
use talkbank_model::alignment::helpers::TierDomain;
use talkbank_model::model::dependent_tier::{Mor, MorTier};
use talkbank_model::model::{
    ChatFile, DependentTier, Line, NonEmptyString, UserDefinedDependentTier,
};

use crate::dependent_tiers::replace_or_add_tier;
use crate::diff::copy_dependent_tiers;
use crate::diff::preserve::TierKind;
use crate::dp_align::{self, AlignResult, MatchMode};
use crate::extract::{self, ExtractedUtterance};
use crate::wer_conform;

/// Status of a compared token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompareStatus {
    /// Word matches between main and gold.
    Match,
    /// Word present in main but not in gold (insertion).
    ExtraMain,
    /// Word present in gold but not in main (deletion).
    ExtraGold,
}

/// A single token in the comparison output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompareToken {
    /// The word text.
    pub text: String,
    /// Uppercased part-of-speech tag when `%mor` data is available.
    pub pos: Option<String>,
    /// Match status.
    pub status: CompareStatus,
}

/// Per-utterance comparison result.
#[derive(Debug, Clone)]
pub struct UtteranceComparison {
    /// Zero-based utterance index in the main file.
    pub utterance_index: usize,
    /// Speaker code.
    pub speaker: String,
    /// Comparison tokens (matches, insertions, deletions).
    pub tokens: Vec<CompareToken>,
}

/// Aggregate comparison metrics.
#[derive(Debug, Clone, PartialEq)]
pub struct CompareMetrics {
    /// Word Error Rate: (insertions + deletions) / total_gold_words.
    pub wer: f64,
    /// 1.0 - wer (clamped to [0, 1]).
    pub accuracy: f64,
    /// Number of matching words.
    pub matches: usize,
    /// Words in main but not in gold.
    pub insertions: usize,
    /// Words in gold but not in main.
    pub deletions: usize,
    /// Total words in the gold transcript (matches + deletions).
    pub total_gold_words: usize,
    /// Total words in the main transcript (matches + insertions).
    pub total_main_words: usize,
    /// Per-POS error breakdown keyed by uppercased POS label.
    pub pos_counts: BTreeMap<String, PosErrorCounts>,
}

/// Per-POS compare counters.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PosErrorCounts {
    /// Number of matching tokens for this POS.
    pub matches: usize,
    /// Number of insertion tokens for this POS.
    pub insertions: usize,
    /// Number of deletion tokens for this POS.
    pub deletions: usize,
}

/// Full comparison bundle.
///
/// This is the internal workflow artifact produced by transcript comparison.
/// It can later support multiple materialization paths (main-annotated output,
/// gold-projected output, metrics sidecars, debugging views) without forcing
/// the compare stage itself to decide the final output shape.
#[derive(Debug, Clone)]
pub struct ComparisonBundle {
    /// Main-anchored per-utterance comparison annotations.
    pub main_utterances: Vec<UtteranceComparison>,
    /// Gold-anchored per-utterance comparison annotations.
    pub gold_utterances: Vec<UtteranceComparison>,
    /// Structural word matches from gold back to the matched main word.
    pub gold_word_matches: Vec<GoldWordMatch>,
    /// Aggregate metrics.
    pub metrics: CompareMetrics,
}

/// Compatibility alias retained while the compare pipeline is refactored toward
/// workflow bundles plus explicit materializers.
pub type CompareResult = ComparisonBundle;

/// Structured serialization errors for compare-owned output artifacts.
#[derive(Debug, thiserror::Error)]
pub enum CompareSerializationError {
    /// `%xsrep` cannot contain an empty token payload.
    #[error(
        "compare serialization produced empty content for xsrep at utterance {utterance_index} token {token_index}"
    )]
    EmptyXsrepToken {
        /// Zero-based utterance index in the source comparison.
        utterance_index: usize,
        /// Zero-based token index within the utterance comparison.
        token_index: usize,
    },
    /// `%xsmor` cannot contain an empty POS payload.
    #[error(
        "compare serialization produced empty content for xsmor at utterance {utterance_index} token {token_index}"
    )]
    EmptyXsmorToken {
        /// Zero-based utterance index in the source comparison.
        utterance_index: usize,
        /// Zero-based token index within the utterance comparison.
        token_index: usize,
    },
    /// Per-POS CSV rows must have a non-empty label.
    #[error("compare serialization produced empty content for compare metrics POS label")]
    EmptyMetricsPosLabel,
    /// A serialized compare tier payload must not collapse to empty text.
    #[error("compare serialization produced empty content for %{label}")]
    EmptyTierContent {
        /// Compare tier label without the leading `%`.
        label: CompareTierLabel,
    },
    /// CSV writer failed while rendering compare metrics.
    #[error("compare CSV serialization failed: {0}")]
    Csv(#[from] csv::Error),
    /// Structured CSV output should always be UTF-8, but convert explicitly.
    #[error("compare CSV output was not valid UTF-8: {0}")]
    Utf8(#[from] std::string::FromUtf8Error),
}

/// Newtype for compare user-defined tier labels.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompareTierLabel(NonEmptyString);

impl CompareTierLabel {
    /// `%xsrep`
    pub fn xsrep() -> Self {
        Self(NonEmptyString::new_unchecked("xsrep"))
    }

    /// `%xsmor`
    pub fn xsmor() -> Self {
        Self(NonEmptyString::new_unchecked("xsmor"))
    }

    fn as_str(&self) -> &str {
        self.0.as_ref()
    }

    fn into_inner(self) -> NonEmptyString {
        self.0
    }
}

impl std::fmt::Display for CompareTierLabel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Newtype for one `%xsrep` token payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompareSurfaceToken(NonEmptyString);

impl CompareSurfaceToken {
    fn new(
        text: &str,
        utterance_index: usize,
        token_index: usize,
    ) -> Result<Self, CompareSerializationError> {
        NonEmptyString::new(text)
            .map(Self)
            .ok_or(CompareSerializationError::EmptyXsrepToken {
                utterance_index,
                token_index,
            })
    }
}

impl WriteChat for CompareSurfaceToken {
    fn write_chat<W: std::fmt::Write>(&self, w: &mut W) -> std::fmt::Result {
        w.write_str(self.0.as_ref())
    }
}

/// Newtype for one `%xsmor` POS payload or per-POS metric key fragment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComparePosLabel(NonEmptyString);

impl ComparePosLabel {
    fn for_tier(
        pos: Option<&str>,
        utterance_index: usize,
        token_index: usize,
    ) -> Result<Self, CompareSerializationError> {
        let raw = pos.unwrap_or("?");
        NonEmptyString::new(raw)
            .map(Self)
            .ok_or(CompareSerializationError::EmptyXsmorToken {
                utterance_index,
                token_index,
            })
    }

    fn for_metrics(raw: &str) -> Result<Self, CompareSerializationError> {
        NonEmptyString::new(raw)
            .map(Self)
            .ok_or(CompareSerializationError::EmptyMetricsPosLabel)
    }

    fn as_str(&self) -> &str {
        self.0.as_ref()
    }
}

impl WriteChat for ComparePosLabel {
    fn write_chat<W: std::fmt::Write>(&self, w: &mut W) -> std::fmt::Result {
        w.write_str(self.0.as_ref())
    }
}

/// Structural prefix marker used in compare user-defined tiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompareTierMarker {
    /// Match: no prefix
    Match,
    /// Main-side extra: `+`
    ExtraMain,
    /// Gold-side extra: `-`
    ExtraGold,
}

impl From<CompareStatus> for CompareTierMarker {
    fn from(value: CompareStatus) -> Self {
        match value {
            CompareStatus::Match => Self::Match,
            CompareStatus::ExtraMain => Self::ExtraMain,
            CompareStatus::ExtraGold => Self::ExtraGold,
        }
    }
}

impl WriteChat for CompareTierMarker {
    fn write_chat<W: std::fmt::Write>(&self, w: &mut W) -> std::fmt::Result {
        match self {
            Self::Match => Ok(()),
            Self::ExtraMain => w.write_char('+'),
            Self::ExtraGold => w.write_char('-'),
        }
    }
}

/// One structured compare-tier item.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompareTierItem<T> {
    /// Whether the token is bare, prefixed with `+`, or prefixed with `-`.
    pub marker: CompareTierMarker,
    /// Typed payload for the token body.
    pub value: T,
}

impl<T: WriteChat> WriteChat for CompareTierItem<T> {
    fn write_chat<W: std::fmt::Write>(&self, w: &mut W) -> std::fmt::Result {
        self.marker.write_chat(w)?;
        self.value.write_chat(w)
    }
}

/// Structured payload for `%xsrep`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct XsrepTierContent {
    /// One entry per compared token.
    pub items: Vec<CompareTierItem<CompareSurfaceToken>>,
}

impl TryFrom<&UtteranceComparison> for XsrepTierContent {
    type Error = CompareSerializationError;

    fn try_from(comparison: &UtteranceComparison) -> Result<Self, Self::Error> {
        let items = comparison
            .tokens
            .iter()
            .enumerate()
            .map(|(token_index, token)| {
                Ok(CompareTierItem {
                    marker: token.status.into(),
                    value: CompareSurfaceToken::new(
                        &token.text,
                        comparison.utterance_index,
                        token_index,
                    )?,
                })
            })
            .collect::<Result<Vec<_>, CompareSerializationError>>()?;
        Ok(Self { items })
    }
}

impl WriteChat for XsrepTierContent {
    fn write_chat<W: std::fmt::Write>(&self, w: &mut W) -> std::fmt::Result {
        for (idx, item) in self.items.iter().enumerate() {
            if idx > 0 {
                w.write_char(' ')?;
            }
            item.write_chat(w)?;
        }
        Ok(())
    }
}

/// Structured payload for `%xsmor`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct XsmorTierContent {
    /// One entry per compared token.
    pub items: Vec<CompareTierItem<ComparePosLabel>>,
}

impl TryFrom<&UtteranceComparison> for XsmorTierContent {
    type Error = CompareSerializationError;

    fn try_from(comparison: &UtteranceComparison) -> Result<Self, Self::Error> {
        let last_index = comparison.tokens.len().saturating_sub(1);
        let items = comparison
            .tokens
            .iter()
            .enumerate()
            .map(|(token_index, token)| {
                let pos = if token_index == last_index && token.pos.as_deref() == Some("PUNCT") {
                    Some(token.text.as_str())
                } else {
                    token.pos.as_deref()
                };
                Ok(CompareTierItem {
                    marker: token.status.into(),
                    value: ComparePosLabel::for_tier(pos, comparison.utterance_index, token_index)?,
                })
            })
            .collect::<Result<Vec<_>, CompareSerializationError>>()?;
        Ok(Self { items })
    }
}

impl WriteChat for XsmorTierContent {
    fn write_chat<W: std::fmt::Write>(&self, w: &mut W) -> std::fmt::Result {
        for (idx, item) in self.items.iter().enumerate() {
            if idx > 0 {
                w.write_char(' ')?;
            }
            item.write_chat(w)?;
        }
        Ok(())
    }
}

/// Structured compare tier ready to cross into an untyped `%x...` CHAT tier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompareUserDefinedTier<T> {
    /// Tier label without `%`.
    pub label: CompareTierLabel,
    /// Structured compare-tier payload.
    pub content: T,
}

impl<T: WriteChat> CompareUserDefinedTier<T> {
    fn into_dependent_tier(self) -> Result<DependentTier, CompareSerializationError> {
        let content_text = self.content.to_chat_string();
        let Some(content) = NonEmptyString::new(&content_text) else {
            return Err(CompareSerializationError::EmptyTierContent { label: self.label });
        };

        Ok(DependentTier::UserDefined(UserDefinedDependentTier {
            label: self.label.into_inner(),
            content,
            span: Span::DUMMY,
        }))
    }
}

/// Structured compare metrics table for CSV output.
#[derive(Debug, Clone, PartialEq)]
pub struct CompareMetricsCsvTable {
    /// Data rows written after the header row.
    pub rows: Vec<CompareMetricsCsvRow>,
}

impl CompareMetricsCsvTable {
    /// Build a structured CSV table from aggregate compare metrics.
    pub fn from_metrics(metrics: &CompareMetrics) -> Result<Self, CompareSerializationError> {
        let mut rows = vec![
            CompareMetricsCsvRow::new(
                CompareMetricName::Wer,
                CompareMetricValue::Decimal(metrics.wer),
            ),
            CompareMetricsCsvRow::new(
                CompareMetricName::Accuracy,
                CompareMetricValue::Decimal(metrics.accuracy),
            ),
            CompareMetricsCsvRow::new(
                CompareMetricName::Matches,
                CompareMetricValue::Count(metrics.matches),
            ),
            CompareMetricsCsvRow::new(
                CompareMetricName::Insertions,
                CompareMetricValue::Count(metrics.insertions),
            ),
            CompareMetricsCsvRow::new(
                CompareMetricName::Deletions,
                CompareMetricValue::Count(metrics.deletions),
            ),
            CompareMetricsCsvRow::new(
                CompareMetricName::TotalGoldWords,
                CompareMetricValue::Count(metrics.total_gold_words),
            ),
            CompareMetricsCsvRow::new(
                CompareMetricName::TotalMainWords,
                CompareMetricValue::Count(metrics.total_main_words),
            ),
        ];

        for (pos, counts) in &metrics.pos_counts {
            let pos = ComparePosLabel::for_metrics(pos)?;
            rows.push(CompareMetricsCsvRow::new(
                CompareMetricName::Pos {
                    pos: pos.clone(),
                    kind: ComparePosMetricKind::Matches,
                },
                CompareMetricValue::Count(counts.matches),
            ));
            rows.push(CompareMetricsCsvRow::new(
                CompareMetricName::Pos {
                    pos: pos.clone(),
                    kind: ComparePosMetricKind::Insertions,
                },
                CompareMetricValue::Count(counts.insertions),
            ));
            rows.push(CompareMetricsCsvRow::new(
                CompareMetricName::Pos {
                    pos: pos.clone(),
                    kind: ComparePosMetricKind::Deletions,
                },
                CompareMetricValue::Count(counts.deletions),
            ));
            rows.push(CompareMetricsCsvRow::new(
                CompareMetricName::Pos {
                    pos,
                    kind: ComparePosMetricKind::Total,
                },
                CompareMetricValue::Count(counts.matches + counts.deletions),
            ));
        }

        Ok(Self { rows })
    }

    /// Serialize the structured compare metrics table with the standard CSV crate.
    pub fn to_csv_string(&self) -> Result<String, CompareSerializationError> {
        let mut writer = csv::WriterBuilder::new()
            .has_headers(false)
            .from_writer(Vec::new());
        writer.write_record([
            CompareCsvHeader::Metric.as_str(),
            CompareCsvHeader::Value.as_str(),
        ])?;
        for row in &self.rows {
            writer.write_record([row.metric.to_csv_field(), row.value.to_csv_field()])?;
        }
        let bytes = writer
            .into_inner()
            .map_err(|err| CompareSerializationError::Csv(err.into_error().into()))?;
        Ok(String::from_utf8(bytes)?)
    }
}

/// One data row in the compare metrics CSV.
#[derive(Debug, Clone, PartialEq)]
pub struct CompareMetricsCsvRow {
    /// Structured metric key.
    pub metric: CompareMetricName,
    /// Structured metric value.
    pub value: CompareMetricValue,
}

impl CompareMetricsCsvRow {
    fn new(metric: CompareMetricName, value: CompareMetricValue) -> Self {
        Self { metric, value }
    }
}

/// CSV header names for compare metrics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompareCsvHeader {
    /// `metric`
    Metric,
    /// `value`
    Value,
}

impl CompareCsvHeader {
    fn as_str(self) -> &'static str {
        match self {
            Self::Metric => "metric",
            Self::Value => "value",
        }
    }
}

/// Structured metric key for compare CSV output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompareMetricName {
    /// Aggregate word error rate.
    Wer,
    /// Aggregate token accuracy.
    Accuracy,
    /// Aggregate exact-match token count.
    Matches,
    /// Aggregate insertion count.
    Insertions,
    /// Aggregate deletion count.
    Deletions,
    /// Aggregate gold/reference token count.
    TotalGoldWords,
    /// Aggregate main/hypothesis token count.
    TotalMainWords,
    /// Per-POS metric row.
    Pos {
        /// POS label rendered in the metric key.
        pos: ComparePosLabel,
        /// Which per-POS aggregate this row carries.
        kind: ComparePosMetricKind,
    },
}

impl CompareMetricName {
    fn to_csv_field(&self) -> String {
        match self {
            Self::Wer => "wer".to_string(),
            Self::Accuracy => "accuracy".to_string(),
            Self::Matches => "matches".to_string(),
            Self::Insertions => "insertions".to_string(),
            Self::Deletions => "deletions".to_string(),
            Self::TotalGoldWords => "total_gold_words".to_string(),
            Self::TotalMainWords => "total_main_words".to_string(),
            Self::Pos { pos, kind } => format!("{}:{}", pos.as_str(), kind.as_str()),
        }
    }
}

/// Structured value for compare CSV output.
#[derive(Debug, Clone, PartialEq)]
pub enum CompareMetricValue {
    /// Fixed-precision decimal metric.
    Decimal(f64),
    /// Nonnegative count metric.
    Count(usize),
}

impl CompareMetricValue {
    fn to_csv_field(&self) -> String {
        match self {
            Self::Decimal(value) => format!("{value:.4}"),
            Self::Count(value) => value.to_string(),
        }
    }
}

/// Per-POS metric subtype in compare CSV output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComparePosMetricKind {
    /// Per-POS exact-match count.
    Matches,
    /// Per-POS insertion count.
    Insertions,
    /// Per-POS deletion count.
    Deletions,
    /// Per-POS gold/reference total.
    Total,
}

impl ComparePosMetricKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Matches => "matches",
            Self::Insertions => "insertions",
            Self::Deletions => "deletions",
            Self::Total => "total",
        }
    }
}

#[derive(Debug, Clone)]
struct FlattenedWordInfo {
    utterance_index: usize,
    word_position: usize,
    compare_position: usize,
    pos: Option<String>,
}

/// A structural match between one gold word slot and one main word slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GoldWordMatch {
    /// Gold utterance containing the matched word.
    pub gold_utterance_index: usize,
    /// Zero-based compared-word position within the gold utterance.
    pub gold_word_position: usize,
    /// Main utterance supplying the matched word.
    pub main_utterance_index: usize,
    /// Zero-based compared-word position within the main utterance.
    pub main_word_position: usize,
}

/// Punctuation and fillers to exclude from comparison (matching BA2 behavior).
///
/// Terminators are recognized via the typed `Terminator` enum so the set
/// stays in lockstep with the grammar. Separators (`,`, `‡`, `„`) are
/// additionally excluded because BA2's compare skipped them too.
fn is_punct_or_filler(word: &str) -> bool {
    static FILLERS: &[&str] = &["um", "uhm", "em", "mhm", "uhhm", "eh", "uh", "hm"];

    let w = word.trim();
    talkbank_model::model::content::Terminator::is_chat_terminator(w)
        || matches!(w, "," | "‡" | "„")
        || FILLERS.contains(&w.to_lowercase().as_str())
}

fn is_punct_pos(pos: Option<&str>) -> bool {
    pos.is_some_and(|value| value.eq_ignore_ascii_case("PUNCT"))
}

/// Apply conform_words per word, returning expanded tokens and an index
/// mapping back to the original word list.
///
/// `mapping[j]` = index into the original `words` list that `conformed[j]`
/// originated from.
fn conform_with_mapping(words: &[String]) -> (Vec<String>, Vec<usize>) {
    let mut conformed = Vec::new();
    let mut mapping = Vec::new();
    for (idx, word) in words.iter().enumerate() {
        let expanded = wer_conform::conform_words(std::slice::from_ref(word));
        for token in expanded {
            conformed.push(token);
            mapping.push(idx);
        }
    }
    (conformed, mapping)
}

/// Find the best local main-token window for one gold utterance.
///
/// This follows the BA2 compare engine's rough-pass strategy:
/// - compare contiguous windows using bag-of-words overlap
/// - only consider windows near the gold utterance length
/// - prefer better overlap, then fewer wasted tokens, then more alignment
///   matches, then the latest window
fn find_best_segment(gold_tokens: &[String], main_tokens: &[String]) -> (usize, usize) {
    if gold_tokens.is_empty() || main_tokens.is_empty() {
        return (0, 0);
    }

    let gold_len = gold_tokens.len();
    let main_len = main_tokens.len();
    let min_window = std::cmp::max(1, gold_len.saturating_sub(2));
    let max_window = std::cmp::min(main_len, gold_len + 2);
    let gold_counts = token_counts(gold_tokens);

    let mut best = (0usize, std::cmp::min(main_len, gold_len));
    let mut best_score = -1.0f64;
    let mut best_waste: Option<usize> = None;
    let mut best_align_matches = 0usize;

    for span in min_window..=max_window {
        for start in 0..=(main_len - span) {
            let end = start + span;
            let overlap = token_overlap(&main_tokens[start..end], &gold_counts);
            let score = overlap as f64 / gold_len as f64;
            let waste = span.saturating_sub(overlap);
            let align_matches = count_alignment_matches(&main_tokens[start..end], gold_tokens);

            if score > best_score
                || (score == best_score && best_waste.is_none_or(|current| waste < current))
                || (score == best_score
                    && Some(waste) == best_waste
                    && align_matches > best_align_matches)
                || (score == best_score
                    && Some(waste) == best_waste
                    && align_matches == best_align_matches
                    && end > best.1)
            {
                best = (start, end);
                best_score = score;
                best_waste = Some(waste);
                best_align_matches = align_matches;
            }
        }
    }

    if best_score <= 0.0 {
        return (0, 0);
    }

    best
}

fn count_alignment_matches(window: &[String], gold_tokens: &[String]) -> usize {
    dp_align::align(window, gold_tokens, MatchMode::CaseInsensitive)
        .into_iter()
        .filter(|item| matches!(item, AlignResult::Match { .. }))
        .count()
}

fn best_rotation(window_tokens: &[String], gold_tokens: &[String]) -> usize {
    if window_tokens.len() <= 1 {
        return 0;
    }

    let mut best_rotation = 0usize;
    let mut best_matches = 0usize;
    for rotation in 0..window_tokens.len() {
        let rotated: Vec<String> = window_tokens[rotation..]
            .iter()
            .chain(window_tokens[..rotation].iter())
            .cloned()
            .collect();
        let matches = count_alignment_matches(&rotated, gold_tokens);
        if matches > best_matches {
            best_matches = matches;
            best_rotation = rotation;
        }
    }
    best_rotation
}

fn token_counts(tokens: &[String]) -> HashMap<&str, usize> {
    let mut counts = HashMap::new();
    for token in tokens {
        *counts.entry(token.as_str()).or_insert(0) += 1;
    }
    counts
}

fn token_overlap(window: &[String], gold_counts: &HashMap<&str, usize>) -> usize {
    let mut window_counts = HashMap::new();
    for token in window {
        *window_counts.entry(token.as_str()).or_insert(0) += 1;
    }

    window_counts
        .iter()
        .map(|(token, count)| std::cmp::min(*count, *gold_counts.get(token).unwrap_or(&0)))
        .sum()
}

/// Compare a main transcript against a gold-standard reference.
///
/// Both inputs are parsed CHAT files. Words are extracted from the Mor
/// domain (excluding punctuation and fillers), normalized via
/// `conform_words`, then aligned with the Hirschberg DP aligner.
///
/// Returns per-utterance comparison annotations and aggregate metrics.
pub fn compare(main_file: &ChatFile, gold_file: &ChatFile) -> ComparisonBundle {
    // 1. Extract words from both files
    let main_utts = extract::extract_words(main_file, TierDomain::Mor);
    let gold_utts = extract::extract_words(gold_file, TierDomain::Mor);

    // 2. Flatten words, filtering punctuation and fillers
    let (main_words, main_info) = flatten_words(main_file, &main_utts);
    let (gold_words, gold_info) = flatten_words(gold_file, &gold_utts);

    // 3. Apply conform with index mapping
    let (conformed_main, main_map) = conform_with_mapping(&main_words);
    let (conformed_gold, gold_map) = conform_with_mapping(&gold_words);

    // 4. Partition conformed gold tokens by utterance so compare can work
    // sequentially, one gold utterance at a time.
    let mut gold_utt_tokens: Vec<Vec<String>> = vec![Vec::new(); gold_utts.len()];
    let mut gold_utt_maps: Vec<Vec<usize>> = vec![Vec::new(); gold_utts.len()];
    for (conformed_idx, token) in conformed_gold.iter().enumerate() {
        let orig_gold_idx = gold_map[conformed_idx];
        let gold_utt_idx = gold_info[orig_gold_idx].utterance_index;
        gold_utt_tokens[gold_utt_idx].push(token.clone());
        gold_utt_maps[gold_utt_idx].push(orig_gold_idx);
    }

    // 5. Align each gold utterance against the best local main window.
    //
    // Matching batchalign2-master matters more here than "fixing" its
    // semantics: compare only aligns inside the selected window and does not
    // surface skipped main tokens that fall outside that window as insertions.
    let mut main_positioned: Vec<Vec<(f64, CompareToken)>> = vec![Vec::new(); main_utts.len()];
    let mut gold_positioned: Vec<Vec<(f64, CompareToken)>> = vec![Vec::new(); gold_utts.len()];
    let mut gold_word_matches = Vec::new();
    let mut metrics = MetricAccumulator::default();
    let mut search_start = 0usize;
    let mut last_global_main_anchor: Option<(usize, usize)> = None;

    for gold_utt_idx in 0..gold_utts.len() {
        let g_tokens = &gold_utt_tokens[gold_utt_idx];
        let g_maps = &gold_utt_maps[gold_utt_idx];
        if g_tokens.is_empty() {
            continue;
        }

        let remaining_main = &conformed_main[search_start..];
        let (win_start, win_end) = find_best_segment(g_tokens, remaining_main);
        let abs_start = search_start + win_start;
        let abs_end = search_start + win_end;

        let window_main = &conformed_main[abs_start..abs_end];
        let window_len = window_main.len();
        let rotation = best_rotation(window_main, g_tokens);
        let rotated_window: Vec<String> = if rotation == 0 {
            window_main.to_vec()
        } else {
            window_main[rotation..]
                .iter()
                .chain(window_main[..rotation].iter())
                .cloned()
                .collect()
        };
        let default_main_anchor = (window_len > 0)
            .then(|| main_map[abs_start + rotation % window_len])
            .map(|orig_idx| {
                let info = &main_info[orig_idx];
                (info.utterance_index, info.word_position)
            });
        let utt_alignment = dp_align::align(&rotated_window, g_tokens, MatchMode::CaseInsensitive);
        let mut local_main_cursor = 0usize;
        let mut local_gold_cursor = 0usize;
        let mut last_gold_word_position: Option<usize> = None;
        let mut local_main_anchor: Option<(usize, usize)> = None;

        for item in utt_alignment {
            match item {
                AlignResult::Match { key, .. } => {
                    let global_main_idx = abs_start + ((local_main_cursor + rotation) % window_len);
                    let orig_main_idx = main_map[global_main_idx];
                    let main_word = &main_info[orig_main_idx];
                    let orig_gold_idx = g_maps[local_gold_cursor];
                    let gold_word = &gold_info[orig_gold_idx];

                    let token = CompareToken {
                        text: key,
                        pos: main_word.pos.clone(),
                        status: CompareStatus::Match,
                    };
                    metrics.record(&token);
                    main_positioned[main_word.utterance_index]
                        .push((main_word.word_position as f64, token.clone()));
                    gold_positioned[gold_utt_idx].push((gold_word.word_position as f64, token));

                    let structural_match = GoldWordMatch {
                        gold_utterance_index: gold_utt_idx,
                        gold_word_position: gold_word.compare_position,
                        main_utterance_index: main_word.utterance_index,
                        main_word_position: main_word.compare_position,
                    };
                    if gold_word_matches.last() != Some(&structural_match) {
                        gold_word_matches.push(structural_match);
                    }

                    local_main_anchor = Some((main_word.utterance_index, main_word.word_position));
                    last_global_main_anchor = local_main_anchor;
                    last_gold_word_position = Some(gold_word.word_position);
                    local_main_cursor += 1;
                    local_gold_cursor += 1;
                }
                AlignResult::ExtraPayload { key, .. } => {
                    let global_main_idx = abs_start + ((local_main_cursor + rotation) % window_len);
                    let orig_main_idx = main_map[global_main_idx];
                    let main_word = &main_info[orig_main_idx];

                    let token = CompareToken {
                        text: key,
                        pos: main_word.pos.clone(),
                        status: CompareStatus::ExtraMain,
                    };
                    metrics.record(&token);
                    main_positioned[main_word.utterance_index]
                        .push((main_word.word_position as f64, token.clone()));
                    gold_positioned[gold_utt_idx].push((
                        last_gold_word_position.map_or(-0.5, |pos| pos as f64 + 0.5),
                        token,
                    ));

                    local_main_anchor = Some((main_word.utterance_index, main_word.word_position));
                    last_global_main_anchor = local_main_anchor;
                    local_main_cursor += 1;
                }
                AlignResult::ExtraReference { key, .. } => {
                    let orig_gold_idx = g_maps[local_gold_cursor];
                    let gold_word = &gold_info[orig_gold_idx];

                    let token = CompareToken {
                        text: key,
                        pos: gold_word.pos.clone(),
                        status: CompareStatus::ExtraGold,
                    };
                    metrics.record(&token);
                    gold_positioned[gold_utt_idx]
                        .push((gold_word.word_position as f64, token.clone()));

                    if let Some((target_utt, target_word_pos)) = local_main_anchor
                        .or(default_main_anchor)
                        .or(last_global_main_anchor)
                        && let Some(target_tokens) = main_positioned.get_mut(target_utt)
                    {
                        target_tokens.push((target_word_pos as f64 + 0.5, token));
                    }

                    last_gold_word_position = Some(gold_word.word_position);
                    local_gold_cursor += 1;
                }
            }
        }

        search_start = abs_end;
    }

    // 6. Append the gold utterance terminator as a PUNCT token so gold-projected
    // `%xsrep` / `%xsmor` lines match batchalign2-master output shape.
    for (gold_utt_idx, terminator) in collect_utterance_terminators(gold_file)
        .into_iter()
        .enumerate()
    {
        let Some(terminator) = terminator else {
            continue;
        };
        gold_positioned[gold_utt_idx].push((
            gold_utt_tokens[gold_utt_idx].len() as f64,
            CompareToken {
                text: terminator,
                pos: Some("PUNCT".to_string()),
                status: CompareStatus::Match,
            },
        ));
    }

    // 7. Stabilize per-utterance token order.
    for tokens in &mut main_positioned {
        tokens.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    }
    for tokens in &mut gold_positioned {
        tokens.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    }

    let main_utterances = build_utterance_comparisons(&main_utts, main_positioned);
    let gold_utterances = build_utterance_comparisons(&gold_utts, gold_positioned);

    ComparisonBundle {
        main_utterances,
        gold_utterances,
        gold_word_matches,
        metrics: metrics.finish(),
    }
}

#[derive(Default)]
struct MetricAccumulator {
    matches: usize,
    insertions: usize,
    deletions: usize,
    pos_counts: BTreeMap<String, PosErrorCounts>,
}

impl MetricAccumulator {
    fn record(&mut self, token: &CompareToken) {
        if token.pos.as_deref() == Some("PUNCT") {
            return;
        }

        match token.status {
            CompareStatus::Match => {
                self.matches += 1;
                self.pos_counts
                    .entry(metric_pos_label(token.pos.as_deref()))
                    .or_default()
                    .matches += 1;
            }
            CompareStatus::ExtraMain => {
                self.insertions += 1;
                self.pos_counts
                    .entry(metric_pos_label(token.pos.as_deref()))
                    .or_default()
                    .insertions += 1;
            }
            CompareStatus::ExtraGold => {
                self.deletions += 1;
                self.pos_counts
                    .entry(metric_pos_label(token.pos.as_deref()))
                    .or_default()
                    .deletions += 1;
            }
        }
    }

    fn finish(self) -> CompareMetrics {
        let total_gold = self.matches + self.deletions;
        let total_main = self.matches + self.insertions;
        let wer = if total_gold > 0 {
            (self.insertions + self.deletions) as f64 / total_gold as f64
        } else {
            0.0
        };
        let accuracy = (1.0 - wer).clamp(0.0, 1.0);

        CompareMetrics {
            wer,
            accuracy,
            matches: self.matches,
            insertions: self.insertions,
            deletions: self.deletions,
            total_gold_words: total_gold,
            total_main_words: total_main,
            pos_counts: self.pos_counts,
        }
    }
}

fn build_utterance_comparisons(
    utterances: &[ExtractedUtterance],
    positioned: Vec<Vec<(f64, CompareToken)>>,
) -> Vec<UtteranceComparison> {
    utterances
        .iter()
        .enumerate()
        .map(|(utt_idx, utt)| UtteranceComparison {
            utterance_index: utt_idx,
            speaker: utt.speaker.as_str().to_string(),
            tokens: positioned[utt_idx]
                .iter()
                .map(|(_, token)| token.clone())
                .collect(),
        })
        .collect()
}

/// Flatten extracted utterances into a word list and info vector.
///
/// Returns:
/// - `words`: cleaned text for each non-punct/non-filler word
/// - `info`: word position and `%mor`-derived metadata for each word
fn flatten_words(
    chat_file: &ChatFile,
    utts: &[ExtractedUtterance],
) -> (Vec<String>, Vec<FlattenedWordInfo>) {
    let mut words = Vec::new();
    let mut info = Vec::new();
    let mor_positions = collect_mor_pos_labels(chat_file);

    for utt in utts {
        let mut compare_position = 0usize;
        for extracted in &utt.words {
            let text = extracted.text.as_str();
            let pos = mor_positions
                .get(utt.utterance_index.0)
                .and_then(|positions| positions.get(extracted.utterance_word_index.0))
                .cloned()
                .flatten();
            if is_punct_or_filler(text) || is_punct_pos(pos.as_deref()) {
                continue;
            }
            words.push(text.to_string());
            info.push(FlattenedWordInfo {
                utterance_index: utt.utterance_index.0,
                word_position: extracted.utterance_word_index.0,
                compare_position,
                pos,
            });
            compare_position += 1;
        }
    }

    (words, info)
}

fn compared_word_counts(utts: &[ExtractedUtterance]) -> Vec<usize> {
    utts.iter()
        .map(|utt| {
            utt.words
                .iter()
                .filter(|word| !is_punct_or_filler(word.text.as_str()))
                .count()
        })
        .collect()
}

fn alignable_word_counts(utts: &[ExtractedUtterance]) -> Vec<usize> {
    utts.iter().map(|utt| utt.words.len()).collect()
}

fn collect_mor_pos_labels(chat_file: &ChatFile) -> Vec<Vec<Option<String>>> {
    let mut utterance_positions = Vec::new();
    for line in &chat_file.lines {
        if let Line::Utterance(utt) = line {
            let mor_positions = utt
                .dependent_tiers
                .iter()
                .find_map(|tier| match tier {
                    DependentTier::Mor(mor) => Some(
                        mor.items
                            .iter()
                            .map(|item| Some(item.main.pos.to_string().to_uppercase()))
                            .collect(),
                    ),
                    _ => None,
                })
                .unwrap_or_default();
            utterance_positions.push(mor_positions);
        }
    }
    utterance_positions
}

fn collect_utterance_terminators(chat_file: &ChatFile) -> Vec<Option<String>> {
    let mut terminators = Vec::new();
    for line in &chat_file.lines {
        if let Line::Utterance(utt) = line {
            terminators.push(
                utt.main
                    .content
                    .terminator
                    .as_ref()
                    .map(|term| term.to_chat_string()),
            );
        }
    }
    terminators
}

fn collect_utterance_terminator_symbols(chat_file: &ChatFile) -> Vec<Option<String>> {
    let mut terminators = Vec::new();
    for line in &chat_file.lines {
        if let Line::Utterance(utt) = line {
            terminators.push(
                utt.main
                    .content
                    .terminator
                    .as_ref()
                    .map(|term| term.to_chat_string()),
            );
        }
    }
    terminators
}

fn collect_mor_items(chat_file: &ChatFile) -> Vec<Vec<Mor>> {
    let mut utterance_items = Vec::new();
    for line in &chat_file.lines {
        if let Line::Utterance(utt) = line {
            let mor_items = utt
                .mor_tier()
                .map(|mor| mor.items.iter().cloned().collect())
                .unwrap_or_default();
            utterance_items.push(mor_items);
        }
    }
    utterance_items
}

fn exact_projection_source(
    bundle: &ComparisonBundle,
    gold_utterance_index: usize,
    main_compared_word_counts: &[usize],
    gold_compared_word_counts: &[usize],
    main_alignable_word_counts: &[usize],
    gold_alignable_word_counts: &[usize],
) -> Option<usize> {
    let gold_word_count = *gold_compared_word_counts.get(gold_utterance_index)?;
    if gold_word_count == 0 {
        return None;
    }

    let matches: Vec<_> = bundle
        .gold_word_matches
        .iter()
        .copied()
        .filter(|item| item.gold_utterance_index == gold_utterance_index)
        .collect();
    if matches.len() != gold_word_count {
        return None;
    }

    let mut gold_positions: Vec<_> = matches.iter().map(|item| item.gold_word_position).collect();
    gold_positions.sort_unstable();
    gold_positions.dedup();
    if gold_positions.len() != gold_word_count {
        return None;
    }

    let mut main_pairs: Vec<_> = matches
        .iter()
        .map(|item| (item.main_utterance_index, item.main_word_position))
        .collect();
    main_pairs.sort_unstable();
    main_pairs.dedup();
    if main_pairs.len() != gold_word_count {
        return None;
    }

    let mut main_utterance_indices: Vec<_> = matches
        .iter()
        .map(|item| item.main_utterance_index)
        .collect();
    main_utterance_indices.sort_unstable();
    main_utterance_indices.dedup();
    if main_utterance_indices.len() != 1 {
        return None;
    }

    let main_utterance_index = main_utterance_indices[0];
    if main_compared_word_counts.get(main_utterance_index).copied() != Some(gold_word_count) {
        return None;
    }
    if main_alignable_word_counts
        .get(main_utterance_index)
        .copied()
        != gold_alignable_word_counts
            .get(gold_utterance_index)
            .copied()
    {
        return None;
    }

    let gold_tokens = bundle.gold_utterances.get(gold_utterance_index)?;
    if gold_tokens
        .tokens
        .iter()
        .any(|token| token.status != CompareStatus::Match)
    {
        return None;
    }

    Some(main_utterance_index)
}

fn build_projected_mor_tier(
    bundle: &ComparisonBundle,
    gold_utterance_index: usize,
    gold_word_count: usize,
    main_mor_items: &[Vec<Mor>],
    gold_mor_items: &[Vec<Mor>],
    gold_terminators: &[Option<String>],
) -> Option<MorTier> {
    if gold_word_count == 0 {
        return None;
    }

    let matches: Vec<_> = bundle
        .gold_word_matches
        .iter()
        .copied()
        .filter(|item| item.gold_utterance_index == gold_utterance_index)
        .collect();
    if matches.is_empty() {
        return None;
    }

    let mut gold_positions: Vec<_> = matches.iter().map(|item| item.gold_word_position).collect();
    gold_positions.sort_unstable();
    gold_positions.dedup();
    if gold_positions.len() != gold_word_count {
        return None;
    }

    let mut main_pairs: Vec<_> = matches
        .iter()
        .map(|item| (item.main_utterance_index, item.main_word_position))
        .collect();
    main_pairs.sort_unstable();
    main_pairs.dedup();
    if main_pairs.len() != matches.len() {
        return None;
    }

    let mut projected = gold_mor_items
        .get(gold_utterance_index)
        .filter(|items| items.len() == gold_word_count)
        .map(|items| items.iter().cloned().map(Some).collect())
        .unwrap_or_else(|| vec![None; gold_word_count]);

    for matched in matches {
        let mor = main_mor_items
            .get(matched.main_utterance_index)?
            .get(matched.main_word_position)?
            .clone();
        *projected.get_mut(matched.gold_word_position)? = Some(mor);
    }

    let items: Vec<Mor> = projected.into_iter().collect::<Option<Vec<_>>>()?;
    Some(
        MorTier::new_mor(items).with_terminator(
            gold_terminators
                .get(gold_utterance_index)
                .cloned()
                .flatten()
                .map(Into::into),
        ),
    )
}

fn replace_or_add_mor_tier(chat_file: &mut ChatFile, utterance_index: usize, mor: MorTier) {
    let mut utterance_count = 0usize;
    for line in chat_file.lines.iter_mut() {
        if let Line::Utterance(utt) = line {
            if utterance_count == utterance_index {
                replace_or_add_tier(&mut utt.dependent_tiers, DependentTier::Mor(mor));
                break;
            }
            utterance_count += 1;
        }
    }
}

/// Project structurally safe `%mor` / `%gra` / `%wor` annotations from main onto gold.
///
/// This keeps compare projection in the CHAT AST:
/// - exact utterance matches copy aligned dependent tiers wholesale
/// - full gold-word coverage without exact utterance identity still projects `%mor`
pub fn project_gold_structurally(
    main_file: &ChatFile,
    gold_file: &ChatFile,
    bundle: &ComparisonBundle,
) -> ChatFile {
    let mut projected = gold_file.clone();
    let main_utts = extract::extract_words(main_file, TierDomain::Mor);
    let gold_utts = extract::extract_words(gold_file, TierDomain::Mor);
    let main_compared_word_counts = compared_word_counts(&main_utts);
    let gold_compared_word_counts = compared_word_counts(&gold_utts);
    let main_alignable_word_counts = alignable_word_counts(&main_utts);
    let gold_alignable_word_counts = alignable_word_counts(&gold_utts);
    let main_mor_items = collect_mor_items(main_file);
    let gold_mor_items = collect_mor_items(gold_file);
    let gold_terminators = collect_utterance_terminator_symbols(gold_file);

    for gold_utterance_index in 0..gold_utts.len() {
        if let Some(main_utterance_index) = exact_projection_source(
            bundle,
            gold_utterance_index,
            &main_compared_word_counts,
            &gold_compared_word_counts,
            &main_alignable_word_counts,
            &gold_alignable_word_counts,
        ) {
            copy_dependent_tiers(
                main_file,
                UtteranceIdx(main_utterance_index),
                &mut projected,
                UtteranceIdx(gold_utterance_index),
                &[TierKind::Mor, TierKind::Gra, TierKind::Wor],
            );
            continue;
        }

        if let Some(projected_mor) = build_projected_mor_tier(
            bundle,
            gold_utterance_index,
            gold_compared_word_counts[gold_utterance_index],
            &main_mor_items,
            &gold_mor_items,
            &gold_terminators,
        ) {
            replace_or_add_mor_tier(&mut projected, gold_utterance_index, projected_mor);
        }
    }

    projected
}

fn metric_pos_label(pos: Option<&str>) -> String {
    pos.unwrap_or("?").to_uppercase()
}

/// Serialize comparison results as a `%xsrep` tier payload.
pub fn format_xsrep(comparison: &UtteranceComparison) -> Result<String, CompareSerializationError> {
    Ok(XsrepTierContent::try_from(comparison)?.to_chat_string())
}

/// Serialize comparison results as a `%xsmor` tier payload.
pub fn format_xsmor(comparison: &UtteranceComparison) -> Result<String, CompareSerializationError> {
    Ok(XsmorTierContent::try_from(comparison)?.to_chat_string())
}

/// Serialize comparison metrics as CSV rows with header.
pub fn format_metrics_csv(metrics: &CompareMetrics) -> Result<String, CompareSerializationError> {
    CompareMetricsCsvTable::from_metrics(metrics)?.to_csv_string()
}

/// Inject comparison results into a CHAT file as `%xsrep` and `%xsmor` tiers.
///
/// For each [`UtteranceComparison`], finds the corresponding utterance in the
/// file (by `utterance_index`) and adds user-defined tiers containing
/// the formatted comparison annotations.
///
/// Uses `replace_or_add_tier` to ensure idempotent injection.
pub fn inject_comparison(
    chat_file: &mut ChatFile,
    utterances: &[UtteranceComparison],
) -> Result<(), CompareSerializationError> {
    let mut utt_line_indices: Vec<usize> = Vec::new();
    for (line_idx, line) in chat_file.lines.iter().enumerate() {
        if matches!(line, Line::Utterance(_)) {
            utt_line_indices.push(line_idx);
        }
    }

    for utt_comparison in utterances {
        if utt_comparison.tokens.is_empty() {
            continue;
        }

        let utt_idx = utt_comparison.utterance_index;
        if utt_idx >= utt_line_indices.len() {
            tracing::warn!(
                utt_idx,
                num_utterances = utt_line_indices.len(),
                "Compare utterance_index out of range"
            );
            continue;
        }

        let line_idx = utt_line_indices[utt_idx];
        let xsrep_tier = CompareUserDefinedTier {
            label: CompareTierLabel::xsrep(),
            content: XsrepTierContent::try_from(utt_comparison)?,
        };
        let xsmor_tier = CompareUserDefinedTier {
            label: CompareTierLabel::xsmor(),
            content: XsmorTierContent::try_from(utt_comparison)?,
        };

        if let Some(Line::Utterance(utt)) = chat_file.lines.get_mut(line_idx) {
            replace_or_add_user_defined_tier(utt, xsrep_tier)?;
            replace_or_add_user_defined_tier(utt, xsmor_tier)?;
        }
    }

    Ok(())
}

fn replace_or_add_user_defined_tier<T: WriteChat>(
    utterance: &mut talkbank_model::model::Utterance,
    tier: CompareUserDefinedTier<T>,
) -> Result<(), CompareSerializationError> {
    let new_tier = tier.into_dependent_tier()?;
    replace_or_add_tier(&mut utterance.dependent_tiers, new_tier);
    Ok(())
}

/// Remove existing `%xsrep` and `%xsmor` tiers from all utterances.
pub fn clear_comparison(chat_file: &mut ChatFile) {
    for line in chat_file.lines.iter_mut() {
        if let Line::Utterance(utt) = line {
            utt.dependent_tiers.retain(|tier| {
                !matches!(
                    tier,
                    DependentTier::UserDefined(ud)
                        if ud.label.as_str() == "xsrep" || ud.label.as_str() == "xsmor"
                )
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use talkbank_model::{ChatFile, ErrorCollector};
    use talkbank_parser::TreeSitterParser;

    fn parse_lenient(
        parser: &TreeSitterParser,
        chat_text: &str,
    ) -> (ChatFile, Vec<talkbank_model::ParseError>) {
        let errors = ErrorCollector::new();
        let chat_file = parser.parse_chat_file_streaming(chat_text, &errors);
        let error_vec = errors.into_vec();
        (chat_file, error_vec)
    }

    /// Build a minimal CHAT file with given utterance lines.
    fn make_chat(utterances: &[(&str, &str)]) -> String {
        let mut lines = vec![
            "@UTF8".to_string(),
            "@Begin".to_string(),
            "@Languages:\teng".to_string(),
            "@Participants:\tCHI Target_Child, MOT Mother".to_string(),
            "@ID:\teng|test|CHI|3;|female|||Target_Child|||".to_string(),
            "@ID:\teng|test|MOT||female|||Mother|||".to_string(),
        ];
        for (speaker, text) in utterances {
            lines.push(format!("*{speaker}:\t{text}"));
        }
        lines.push("@End".to_string());
        lines.join("\n")
    }

    #[test]
    fn identical_transcripts() {
        let parser = TreeSitterParser::new().unwrap();
        let chat = make_chat(&[("CHI", "hello world ."), ("MOT", "good morning .")]);
        let (main_file, _) = parse_lenient(&parser, &chat);
        let (gold_file, _) = parse_lenient(&parser, &chat);

        let result = compare(&main_file, &gold_file);
        assert_eq!(result.metrics.wer, 0.0);
        assert_eq!(result.metrics.accuracy, 1.0);
        assert_eq!(result.metrics.matches, 4);
        assert_eq!(result.metrics.insertions, 0);
        assert_eq!(result.metrics.deletions, 0);
        assert_eq!(result.metrics.total_gold_words, 4);
        assert_eq!(result.metrics.total_main_words, 4);
    }

    #[test]
    fn single_substitution() {
        let parser = TreeSitterParser::new().unwrap();
        let main = make_chat(&[("CHI", "hello earth .")]);
        let gold = make_chat(&[("CHI", "hello world .")]);
        let (main_file, _) = parse_lenient(&parser, &main);
        let (gold_file, _) = parse_lenient(&parser, &gold);

        let result = compare(&main_file, &gold_file);
        // Compare aligns inside the best local window and does not surface
        // skipped main tokens outside that window as insertions.
        assert!(result.metrics.wer > 0.0);
        assert_eq!(result.metrics.matches, 1); // "hello" matches
        assert_eq!(result.metrics.insertions, 0); // skipped "earth"
        assert_eq!(result.metrics.deletions, 1); // "world"
    }

    #[test]
    fn extra_word_in_main() {
        let parser = TreeSitterParser::new().unwrap();
        let main = make_chat(&[("CHI", "hello big world .")]);
        let gold = make_chat(&[("CHI", "hello world .")]);
        let (main_file, _) = parse_lenient(&parser, &main);
        let (gold_file, _) = parse_lenient(&parser, &gold);

        let result = compare(&main_file, &gold_file);
        assert_eq!(result.metrics.matches, 2); // "hello", "world"
        assert_eq!(result.metrics.insertions, 1); // "big"
        assert_eq!(result.metrics.deletions, 0);
        assert_eq!(result.metrics.total_gold_words, 2);
        assert_eq!(result.metrics.total_main_words, 3);
    }

    #[test]
    fn missing_word_in_main() {
        let parser = TreeSitterParser::new().unwrap();
        let main = make_chat(&[("CHI", "hello .")]);
        let gold = make_chat(&[("CHI", "hello world .")]);
        let (main_file, _) = parse_lenient(&parser, &main);
        let (gold_file, _) = parse_lenient(&parser, &gold);

        let result = compare(&main_file, &gold_file);
        assert_eq!(result.metrics.matches, 1); // "hello"
        assert_eq!(result.metrics.insertions, 0);
        assert_eq!(result.metrics.deletions, 1); // "world"
        assert_eq!(result.metrics.total_gold_words, 2);
        assert_eq!(result.metrics.total_main_words, 1);
    }

    #[test]
    fn empty_main() {
        let parser = TreeSitterParser::new().unwrap();
        // Main has an utterance but no content words (just terminator)
        let main = make_chat(&[("CHI", ".")]);
        let gold = make_chat(&[("CHI", "hello world .")]);
        let (main_file, _) = parse_lenient(&parser, &main);
        let (gold_file, _) = parse_lenient(&parser, &gold);

        let result = compare(&main_file, &gold_file);
        assert_eq!(result.metrics.matches, 0);
        assert_eq!(result.metrics.deletions, 2);
        assert_eq!(result.metrics.wer, 1.0);
    }

    #[test]
    fn empty_gold() {
        let parser = TreeSitterParser::new().unwrap();
        let main = make_chat(&[("CHI", "hello world .")]);
        let gold = make_chat(&[("CHI", ".")]);
        let (main_file, _) = parse_lenient(&parser, &main);
        let (gold_file, _) = parse_lenient(&parser, &gold);

        let result = compare(&main_file, &gold_file);
        assert_eq!(result.metrics.matches, 0);
        assert_eq!(result.metrics.insertions, 0);
        assert_eq!(result.metrics.total_gold_words, 0);
        assert_eq!(result.metrics.wer, 0.0); // no gold words => wer=0
    }

    #[test]
    fn case_insensitive_matching() {
        let parser = TreeSitterParser::new().unwrap();
        let main = make_chat(&[("CHI", "Hello World .")]);
        let gold = make_chat(&[("CHI", "hello world .")]);
        let (main_file, _) = parse_lenient(&parser, &main);
        let (gold_file, _) = parse_lenient(&parser, &gold);

        let result = compare(&main_file, &gold_file);
        assert_eq!(result.metrics.wer, 0.0);
        assert_eq!(result.metrics.matches, 2);
    }

    #[test]
    fn conform_normalizes_contractions() {
        let parser = TreeSitterParser::new().unwrap();
        // "he's" should be expanded to "he is" by conform_words
        let main = make_chat(&[("CHI", "he's going .")]);
        let gold = make_chat(&[("CHI", "he is going .")]);
        let (main_file, _) = parse_lenient(&parser, &main);
        let (gold_file, _) = parse_lenient(&parser, &gold);

        let result = compare(&main_file, &gold_file);
        // After conform: main = ["he", "is", "going"], gold = ["he", "is", "going"]
        assert_eq!(result.metrics.wer, 0.0);
    }

    #[test]
    fn multiple_utterances() {
        let parser = TreeSitterParser::new().unwrap();
        let main = make_chat(&[("CHI", "hello ."), ("MOT", "goodbye .")]);
        let gold = make_chat(&[("CHI", "hello ."), ("MOT", "goodbye .")]);
        let (main_file, _) = parse_lenient(&parser, &main);
        let (gold_file, _) = parse_lenient(&parser, &gold);

        let result = compare(&main_file, &gold_file);
        assert_eq!(result.metrics.wer, 0.0);
        assert_eq!(result.metrics.matches, 2);
        assert_eq!(result.main_utterances.len(), 2);
        assert_eq!(result.gold_utterances.len(), 2);
    }

    #[test]
    fn xsrep_tier_content_serializes_through_write_chat() {
        let utt = UtteranceComparison {
            utterance_index: 0,
            speaker: "CHI".to_string(),
            tokens: vec![
                CompareToken {
                    text: "hello".to_string(),
                    pos: Some("INTJ".to_string()),
                    status: CompareStatus::Match,
                },
                CompareToken {
                    text: "big".to_string(),
                    pos: Some("ADJ".to_string()),
                    status: CompareStatus::ExtraMain,
                },
                CompareToken {
                    text: "world".to_string(),
                    pos: Some("NOUN".to_string()),
                    status: CompareStatus::Match,
                },
                CompareToken {
                    text: "today".to_string(),
                    pos: Some("NOUN".to_string()),
                    status: CompareStatus::ExtraGold,
                },
            ],
        };
        let xsrep = XsrepTierContent::try_from(&utt).expect("xsrep tier");
        assert_eq!(xsrep.to_chat_string(), "hello +big world -today");
    }

    #[test]
    fn xsmor_tier_content_serializes_through_write_chat() {
        let utt = UtteranceComparison {
            utterance_index: 0,
            speaker: "CHI".to_string(),
            tokens: vec![
                CompareToken {
                    text: "hello".to_string(),
                    pos: Some("INTJ".to_string()),
                    status: CompareStatus::Match,
                },
                CompareToken {
                    text: "big".to_string(),
                    pos: Some("ADJ".to_string()),
                    status: CompareStatus::ExtraMain,
                },
                CompareToken {
                    text: "today".to_string(),
                    pos: None,
                    status: CompareStatus::ExtraGold,
                },
            ],
        };
        let xsmor = XsmorTierContent::try_from(&utt).expect("xsmor tier");
        assert_eq!(xsmor.to_chat_string(), "INTJ +ADJ -?");
    }

    #[test]
    fn xsmor_serializes_final_punctuation_as_surface_delimiter() {
        let utt = UtteranceComparison {
            utterance_index: 0,
            speaker: "CHI".to_string(),
            tokens: vec![
                CompareToken {
                    text: "hello".to_string(),
                    pos: Some("INTJ".to_string()),
                    status: CompareStatus::Match,
                },
                CompareToken {
                    text: ".".to_string(),
                    pos: Some("PUNCT".to_string()),
                    status: CompareStatus::Match,
                },
            ],
        };
        let xsmor = XsmorTierContent::try_from(&utt).expect("xsmor tier");
        assert_eq!(xsmor.to_chat_string(), "INTJ .");
    }

    #[test]
    fn compare_metrics_csv_table_serializes_with_csv_writer() {
        let metrics = CompareMetrics {
            wer: 0.25,
            accuracy: 0.75,
            matches: 3,
            insertions: 1,
            deletions: 0,
            total_gold_words: 3,
            total_main_words: 4,
            pos_counts: BTreeMap::from([(
                "NOUN".to_string(),
                PosErrorCounts {
                    matches: 2,
                    insertions: 1,
                    deletions: 0,
                },
            )]),
        };
        let csv = CompareMetricsCsvTable::from_metrics(&metrics)
            .expect("table")
            .to_csv_string()
            .expect("csv");
        assert!(csv.contains("wer,0.2500"));
        assert!(csv.contains("accuracy,0.7500"));
        assert!(csv.contains("matches,3"));
        assert!(csv.contains("insertions,1"));
        assert!(csv.contains("deletions,0"));
        assert!(csv.contains("NOUN:matches,2"));
        assert!(csv.contains("NOUN:insertions,1"));
    }

    #[test]
    fn wer_computation_is_correct() {
        // 2 matches, 1 insertion, 1 deletion out of 3 gold words
        let metrics = CompareMetrics {
            wer: 0.0,      // will be computed
            accuracy: 0.0, // will be computed
            matches: 2,
            insertions: 1,
            deletions: 1,
            total_gold_words: 3, // matches + deletions
            total_main_words: 3, // matches + insertions
            pos_counts: BTreeMap::new(),
        };
        // WER = (ins + del) / total_gold = 2/3 ≈ 0.6667
        let expected_wer = 2.0 / 3.0;
        let _ = metrics;

        // Test via actual compare
        let parser = TreeSitterParser::new().unwrap();
        let main = make_chat(&[("CHI", "hello big world .")]);
        let gold = make_chat(&[("CHI", "hello world today .")]);
        let (main_file, _) = parse_lenient(&parser, &main);
        let (gold_file, _) = parse_lenient(&parser, &gold);

        let result = compare(&main_file, &gold_file);
        // main: hello, big, world
        // gold: hello, world, today
        // align: hello=match, big=extra_main, world=match, today=extra_gold
        assert_eq!(result.metrics.matches, 2);
        assert_eq!(result.metrics.insertions, 1);
        assert_eq!(result.metrics.deletions, 1);
        assert!((result.metrics.wer - expected_wer).abs() < 0.001);
    }

    #[test]
    fn is_punct_or_filler_works() {
        assert!(is_punct_or_filler("."));
        assert!(is_punct_or_filler("?"));
        assert!(is_punct_or_filler("!"));
        assert!(is_punct_or_filler("+/."));
        assert!(is_punct_or_filler("um"));
        assert!(is_punct_or_filler("uh"));
        assert!(!is_punct_or_filler("hello"));
        assert!(!is_punct_or_filler("world"));
    }

    #[test]
    fn conform_with_mapping_tracks_indices() {
        let words: Vec<String> = vec!["he's".to_string(), "going".to_string()];
        let (conformed, mapping) = conform_with_mapping(&words);
        // "he's" -> ["he", "is"], "going" -> ["going"]
        assert_eq!(conformed, vec!["he", "is", "going"]);
        assert_eq!(mapping, vec![0, 0, 1]);
    }

    #[test]
    fn inject_comparison_adds_xsrep_tiers() {
        let parser = TreeSitterParser::new().unwrap();
        let main = make_chat(&[("CHI", "hello big world .")]);
        let gold = make_chat(&[("CHI", "hello world .")]);
        let (mut main_file, _) = parse_lenient(&parser, &main);
        let (gold_file, _) = parse_lenient(&parser, &gold);

        let result = compare(&main_file, &gold_file);
        inject_comparison(&mut main_file, &result.main_utterances).expect("inject comparison");

        // Find the utterance and check it has an %xsrep tier
        let serialized = main_file.to_chat_string();
        assert!(
            serialized.contains("%xsrep:"),
            "Output should contain %xsrep tier"
        );
        assert!(
            serialized.contains("+big"),
            "Should mark 'big' as extra_main"
        );
        assert!(
            serialized.contains("%xsmor:"),
            "Output should contain %xsmor tier"
        );
    }

    #[test]
    fn clear_comparison_removes_compare_tiers() {
        let parser = TreeSitterParser::new().unwrap();
        let main = make_chat(&[("CHI", "hello world .")]);
        let gold = make_chat(&[("CHI", "hello world .")]);
        let (mut main_file, _) = parse_lenient(&parser, &main);
        let (gold_file, _) = parse_lenient(&parser, &gold);

        let result = compare(&main_file, &gold_file);
        inject_comparison(&mut main_file, &result.main_utterances).expect("inject comparison");

        // Verify xsrep was added
        let serialized = main_file.to_chat_string();
        assert!(serialized.contains("%xsrep:"));
        assert!(serialized.contains("%xsmor:"));

        // Clear and verify removal
        clear_comparison(&mut main_file);
        let serialized = main_file.to_chat_string();
        assert!(!serialized.contains("%xsrep:"));
        assert!(!serialized.contains("%xsmor:"));
    }

    #[test]
    fn inject_comparison_idempotent() {
        let parser = TreeSitterParser::new().unwrap();
        let main = make_chat(&[("CHI", "hello big world .")]);
        let gold = make_chat(&[("CHI", "hello world .")]);
        let (mut main_file, _) = parse_lenient(&parser, &main);
        let (gold_file, _) = parse_lenient(&parser, &gold);

        let result = compare(&main_file, &gold_file);
        inject_comparison(&mut main_file, &result.main_utterances).expect("inject comparison");
        let first = main_file.to_chat_string();

        // Inject again — should produce the same output (replace, not duplicate)
        inject_comparison(&mut main_file, &result.main_utterances).expect("inject comparison");
        let second = main_file.to_chat_string();
        assert_eq!(first, second);
    }

    #[test]
    fn format_metrics_csv_has_header() {
        let metrics = CompareMetrics {
            wer: 0.25,
            accuracy: 0.75,
            matches: 3,
            insertions: 1,
            deletions: 0,
            total_gold_words: 3,
            total_main_words: 4,
            pos_counts: BTreeMap::new(),
        };
        let csv = CompareMetricsCsvTable::from_metrics(&metrics)
            .expect("table")
            .to_csv_string()
            .expect("csv");
        assert!(csv.starts_with("metric,value\n"));
        assert!(csv.contains("wer,0.2500"));
    }

    #[test]
    fn inject_comparison_rejects_empty_compare_tokens() {
        let parser = TreeSitterParser::new().unwrap();
        let main = make_chat(&[("CHI", "hello .")]);
        let (mut main_file, _) = parse_lenient(&parser, &main);

        let utterances = vec![UtteranceComparison {
            utterance_index: 0,
            speaker: "CHI".to_string(),
            tokens: vec![CompareToken {
                text: String::new(),
                pos: Some("INTJ".to_string()),
                status: CompareStatus::Match,
            }],
        }];

        let err =
            inject_comparison(&mut main_file, &utterances).expect_err("should reject empty token");
        assert!(err.to_string().contains("empty content"));
    }

    #[test]
    fn compare_uses_mor_pos_for_xsmor_output() {
        let parser = TreeSitterParser::new().unwrap();
        let main = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n@ID:\teng|test|CHI|||||Target_Child|||\n*CHI:\thello world .\n%mor:\tintj|hello noun|world .\n@End\n";
        let gold = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n@ID:\teng|test|CHI|||||Target_Child|||\n*CHI:\thello world .\n@End\n";
        let (main_file, _) = parse_lenient(&parser, main);
        let (gold_file, _) = parse_lenient(&parser, gold);

        let result = compare(&main_file, &gold_file);
        assert_eq!(
            XsmorTierContent::try_from(&result.main_utterances[0])
                .expect("xsmor tier")
                .to_chat_string(),
            "INTJ NOUN"
        );
        assert_eq!(result.metrics.pos_counts["INTJ"].matches, 1);
        assert_eq!(result.metrics.pos_counts["NOUN"].matches, 1);
    }

    #[test]
    fn compare_ignores_pos_punct_even_when_surface_is_not_punctuation() {
        let parser = TreeSitterParser::new().unwrap();
        let main = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n@ID:\teng|test|CHI|||||Target_Child|||\n*CHI:\thello comma .\n%mor:\tintj|hello PUNCT|comma .\n@End\n";
        let gold = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n@ID:\teng|test|CHI|||||Target_Child|||\n*CHI:\thello .\n@End\n";
        let (main_file, _) = parse_lenient(&parser, main);
        let (gold_file, _) = parse_lenient(&parser, gold);

        let result = compare(&main_file, &gold_file);
        assert_eq!(result.metrics.matches, 1);
        assert_eq!(result.metrics.insertions, 0);
        assert_eq!(result.metrics.deletions, 0);
        assert_eq!(result.metrics.wer, 0.0);
    }

    #[test]
    fn compare_rotation_recovers_cyclic_window_matches() {
        let parser = TreeSitterParser::new().unwrap();
        let main = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n@ID:\teng|test|CHI|||||Target_Child|||\n*CHI:\tworld hello .\n%mor:\tnoun|world intj|hello .\n@End\n";
        let gold = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n@ID:\teng|test|CHI|||||Target_Child|||\n*CHI:\thello world .\n@End\n";
        let (main_file, _) = parse_lenient(&parser, main);
        let (gold_file, _) = parse_lenient(&parser, gold);

        let result = compare(&main_file, &gold_file);
        assert_eq!(result.metrics.matches, 2);
        assert_eq!(result.metrics.insertions, 0);
        assert_eq!(result.metrics.deletions, 0);
        assert_eq!(
            XsrepTierContent::try_from(&result.gold_utterances[0])
                .expect("xsrep tier")
                .to_chat_string(),
            "hello world ."
        );
    }

    #[test]
    fn batchalign2_master_simple_gold_projection_shape() {
        let parser = TreeSitterParser::new().unwrap();
        let main = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n@ID:\teng|test|CHI|||||Target_Child|||\n*CHI:\thello big world .\n%mor:\tintj|hello adj|big noun|world .\n@End\n";
        let gold = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n@ID:\teng|test|CHI|||||Target_Child|||\n*CHI:\thello world today .\n@End\n";
        let (main_file, _) = parse_lenient(&parser, main);
        let (gold_file, _) = parse_lenient(&parser, gold);

        let result = compare(&main_file, &gold_file);
        assert_eq!(
            XsrepTierContent::try_from(&result.gold_utterances[0])
                .expect("xsrep tier")
                .to_chat_string(),
            "hello +big world -today ."
        );
        assert_eq!(
            XsmorTierContent::try_from(&result.gold_utterances[0])
                .expect("xsmor tier")
                .to_chat_string(),
            "INTJ +ADJ NOUN -? ."
        );
        assert_eq!(result.metrics.matches, 2);
        assert_eq!(result.metrics.insertions, 1);
        assert_eq!(result.metrics.deletions, 1);
        assert!((result.metrics.wer - (2.0 / 3.0)).abs() < 0.001);
        assert_eq!(result.metrics.pos_counts["ADJ"].insertions, 1);
        assert_eq!(result.metrics.pos_counts["?"].deletions, 1);
    }

    #[test]
    fn batchalign2_master_windowed_alignment_ignores_skipped_prefix_tokens() {
        let parser = TreeSitterParser::new().unwrap();
        let main = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n@ID:\teng|test|CHI|||||Target_Child|||\n*CHI:\tdog dog the dog .\n%mor:\tnoun|dog noun|dog det|the noun|dog .\n@End\n";
        let gold = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n@ID:\teng|test|CHI|||||Target_Child|||\n*CHI:\tthe dog .\n@End\n";
        let (main_file, _) = parse_lenient(&parser, main);
        let (gold_file, _) = parse_lenient(&parser, gold);

        let result = compare(&main_file, &gold_file);
        assert_eq!(result.metrics.matches, 2);
        assert_eq!(result.metrics.insertions, 0);
        assert_eq!(result.metrics.deletions, 0);
        assert_eq!(result.metrics.wer, 0.0);
        assert_eq!(
            XsrepTierContent::try_from(&result.gold_utterances[0])
                .expect("xsrep tier")
                .to_chat_string(),
            "the dog ."
        );
        assert_eq!(
            XsmorTierContent::try_from(&result.gold_utterances[0])
                .expect("xsmor tier")
                .to_chat_string(),
            "DET NOUN ."
        );
    }

    #[test]
    fn batchalign2_master_multi_utterance_compare_metrics() {
        let parser = TreeSitterParser::new().unwrap();
        let main = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n@ID:\teng|test|CHI|||||Target_Child|||\n*CHI:\tone fish two fish .\n%mor:\tnum|one noun|fish num|two noun|fish .\n*CHI:\tred fish blue fish .\n%mor:\tadj|red noun|fish adj|blue noun|fish .\n@End\n";
        let gold = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n@ID:\teng|test|CHI|||||Target_Child|||\n*CHI:\tone fish fish .\n*CHI:\tred fish green fish .\n@End\n";
        let (main_file, _) = parse_lenient(&parser, main);
        let (gold_file, _) = parse_lenient(&parser, gold);

        let result = compare(&main_file, &gold_file);
        assert_eq!(
            XsrepTierContent::try_from(&result.gold_utterances[0])
                .expect("xsrep tier")
                .to_chat_string(),
            "one fish +two fish ."
        );
        assert_eq!(
            XsmorTierContent::try_from(&result.gold_utterances[0])
                .expect("xsmor tier")
                .to_chat_string(),
            "NUM NOUN +NUM NOUN ."
        );
        assert_eq!(
            XsrepTierContent::try_from(&result.gold_utterances[1])
                .expect("xsrep tier")
                .to_chat_string(),
            "red fish -green +blue fish ."
        );
        assert_eq!(
            XsmorTierContent::try_from(&result.gold_utterances[1])
                .expect("xsmor tier")
                .to_chat_string(),
            "ADJ NOUN -? +ADJ NOUN ."
        );
        assert_eq!(result.metrics.matches, 6);
        assert_eq!(result.metrics.insertions, 2);
        assert_eq!(result.metrics.deletions, 1);
        assert!((result.metrics.wer - (3.0 / 7.0)).abs() < 0.001);
    }

    #[test]
    fn gold_anchored_projection_attaches_diff_to_gold_transcript() {
        let parser = TreeSitterParser::new().unwrap();
        let main = make_chat(&[("CHI", "hello big world .")]);
        let gold = make_chat(&[("CHI", "hello world today .")]);
        let (main_file, _) = parse_lenient(&parser, &main);
        let (mut gold_file, _) = parse_lenient(&parser, &gold);

        let result = compare(&main_file, &gold_file);
        inject_comparison(&mut gold_file, &result.gold_utterances).expect("inject comparison");

        let serialized = gold_file.to_chat_string();
        assert!(serialized.contains("*CHI:\thello world today ."));
        assert!(serialized.contains("%xsrep:\thello +big world -today"));
        assert!(serialized.contains("%xsmor:"));
    }
}
