use std::collections::BTreeMap;

use super::language::{
    Gold, InsertionAttribution, LanguageBucket, LanguageScores, LanguageTally, LanguagedWord, Main,
    Rate, SidedLanguage, SiteUtterances, UtteranceIndex, WordLanguage,
};
use super::model::{CompareStatus, CompareToken, PosErrorCounts};
use super::serialize::{ComparePosLabel, CompareSerializationError};

/// Immutable aggregate scores from one comparison. Counts and rates have one owner.
#[derive(Debug, Clone, PartialEq)]
pub struct CompareMetrics {
    displacement_edits: usize,
    pos_counts: BTreeMap<String, PosErrorCounts>,
    languages: LanguageScores,
}

impl CompareMetrics {
    /// Matched alignment tokens.
    pub fn matches(&self) -> usize {
        self.languages.matches()
    }
    /// Unmatched hypothesis tokens, including substitution partners.
    pub fn insertions(&self) -> usize {
        self.languages.unmatched_main_words()
    }
    /// Unmatched reference tokens, including substitution partners.
    pub fn deletions(&self) -> usize {
        self.languages.unmatched_gold_words()
    }
    /// Reference token count.
    pub fn total_gold_words(&self) -> usize {
        self.matches() + self.deletions()
    }
    /// Hypothesis token count.
    pub fn total_main_words(&self) -> usize {
        self.matches() + self.insertions()
    }
    /// Legacy WER: substitutions count as both insertion and deletion.
    pub fn wer(&self) -> f64 {
        self.legacy_rate(self.insertions() + self.deletions())
    }
    /// Error rate after cancelling displacement within gold utterances.
    pub fn cwer(&self) -> f64 {
        self.legacy_rate(self.displacement_edits)
    }
    /// Legacy accuracy, clamped to zero through one.
    pub fn accuracy(&self) -> f64 {
        (1.0 - self.wer()).clamp(0.0, 1.0)
    }
    /// Immutable per-POS counts.
    pub fn pos_counts(&self) -> &BTreeMap<String, PosErrorCounts> {
        &self.pos_counts
    }
    /// Immutable language scores, the source of aggregate counts.
    pub fn languages(&self) -> &LanguageScores {
        &self.languages
    }
    /// Error rate with substitutions paired within alignment gaps.
    pub fn wer_with_substitutions(&self) -> Rate {
        self.languages.wer_with_substitutions()
    }

    fn legacy_rate(&self, edits: usize) -> f64 {
        match self.total_gold_words() {
            0 => 0.0,
            words => edits as f64 / words as f64,
        }
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
                CompareMetricValue::Decimal(metrics.wer()),
            ),
            CompareMetricsCsvRow::new(
                CompareMetricName::Cwer,
                CompareMetricValue::Decimal(metrics.cwer()),
            ),
            CompareMetricsCsvRow::new(
                CompareMetricName::Accuracy,
                CompareMetricValue::Decimal(metrics.accuracy()),
            ),
            CompareMetricsCsvRow::new(
                CompareMetricName::Matches,
                CompareMetricValue::Count(metrics.matches()),
            ),
            CompareMetricsCsvRow::new(
                CompareMetricName::Insertions,
                CompareMetricValue::Count(metrics.insertions()),
            ),
            CompareMetricsCsvRow::new(
                CompareMetricName::Deletions,
                CompareMetricValue::Count(metrics.deletions()),
            ),
            CompareMetricsCsvRow::new(
                CompareMetricName::TotalGoldWords,
                CompareMetricValue::Count(metrics.total_gold_words()),
            ),
            CompareMetricsCsvRow::new(
                CompareMetricName::TotalMainWords,
                CompareMetricValue::Count(metrics.total_main_words()),
            ),
        ];

        for (pos, counts) in metrics.pos_counts() {
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

        // After the per-POS rows, so the columns the consolidated `compare.csv`
        // already had keep their positions for readers that go by position.
        metrics.languages().write_csv_rows(&mut rows);

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
    pub(in crate::compare) fn new(metric: CompareMetricName, value: CompareMetricValue) -> Self {
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
    /// Aggregate order-insensitive word error rate.
    Cwer,
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
    /// Deletions paired with an insertion between the same two matches.
    Substitutions,
    /// `(substitutions + deletions + insertions) / gold words`.
    WerWithSubstitutions,
    /// Unpaired insertions against a gold with no compared words.
    UnattributedInsertions,
    /// Per-gold-language error row.
    Language {
        /// The gold language the row is for.
        language: LanguageBucket,
        /// Which count or rate the row carries.
        kind: LanguageMetricKind,
    },
    /// An agreement total or accuracy.
    Agreement {
        /// Utterances or words.
        scope: AgreementScope,
        /// Which total the row carries.
        kind: AgreementMetricKind,
    },
    /// How often a gold language was given as a main language.
    Confusion {
        /// Utterances or words.
        scope: AgreementScope,
        /// The gold language.
        gold: LanguageBucket,
        /// The main language.
        main: LanguageBucket,
    },
    /// Gold utterances of one language that could not be placed.
    UtteranceLanguageUnplaced {
        /// The gold language.
        language: LanguageBucket,
    },
    /// A code-switch detection count or rate.
    Switch(SwitchMetricKind),
    /// Per-POS metric row.
    Pos {
        /// POS label rendered in the metric key.
        pos: ComparePosLabel,
        /// Which per-POS aggregate this row carries.
        kind: ComparePosMetricKind,
    },
}

impl CompareMetricName {
    /// The field exactly as the metrics CSV writes it.
    pub fn to_csv_field(&self) -> String {
        match self {
            Self::Wer => "wer".to_string(),
            Self::Cwer => "cwer".to_string(),
            Self::Accuracy => "accuracy".to_string(),
            Self::Matches => "matches".to_string(),
            Self::Insertions => "insertions".to_string(),
            Self::Deletions => "deletions".to_string(),
            Self::TotalGoldWords => "total_gold_words".to_string(),
            Self::TotalMainWords => "total_main_words".to_string(),
            Self::Substitutions => "substitutions".to_string(),
            Self::WerWithSubstitutions => "wer_with_substitutions".to_string(),
            Self::UnattributedInsertions => "unattributed_insertions".to_string(),
            Self::Language { language, kind } => {
                format!("language:{}:{}", language.csv_key(), kind.as_str())
            }
            Self::Agreement { scope, kind } => format!("{}:{}", scope.as_str(), kind.as_str()),
            Self::Confusion { scope, gold, main } => format!(
                "{}:gold={}:main={}",
                scope.as_str(),
                gold.csv_key(),
                main.csv_key()
            ),
            Self::UtteranceLanguageUnplaced { language } => format!(
                "{}:unplaced:{}",
                AgreementScope::Utterance.as_str(),
                language.csv_key()
            ),
            Self::Switch(kind) => format!("switch:{}", kind.as_str()),
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
    /// A rate that may have had nothing to divide by, printed `NA` then.
    Rate(Rate),
}

impl CompareMetricValue {
    /// The field exactly as the metrics CSV writes it.
    pub fn to_csv_field(&self) -> String {
        match self {
            Self::Decimal(value) | Self::Rate(Rate::Defined(value)) => format!("{value:.4}"),
            Self::Count(value) => value.to_string(),
            Self::Rate(Rate::NoDenominator) => "NA".to_string(),
        }
    }
}

/// Per-gold-language row subtype.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LanguageMetricKind {
    /// Gold words in the language.
    GoldWords,
    /// Matched gold words.
    Matches,
    /// Substituted gold words.
    Substitutions,
    /// Deleted gold words.
    Deletions,
    /// Unpaired insertions attributed to the language.
    Insertions,
    /// The language's substitution-paired error rate.
    WerWithSubstitutions,
}

impl LanguageMetricKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::GoldWords => "gold_words",
            Self::Matches => "matches",
            Self::Substitutions => "substitutions",
            Self::Deletions => "deletions",
            Self::Insertions => "insertions",
            Self::WerWithSubstitutions => "wer_with_substitutions",
        }
    }
}

/// What a language agreement row is scored over.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgreementScope {
    /// Placed gold utterances.
    Utterance,
    /// Matched words.
    Word,
}

impl AgreementScope {
    fn as_str(self) -> &'static str {
        match self {
            Self::Utterance => "utterance_language",
            Self::Word => "word_language",
        }
    }
}

/// Agreement row subtype.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgreementMetricKind {
    /// Items scored.
    Scored,
    /// Items whose languages agree.
    Agreeing,
    /// Items where either language is unresolved or ambiguous.
    Indeterminate,
    /// `agreeing / scored`.
    Accuracy,
}

impl AgreementMetricKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Scored => "scored",
            Self::Agreeing => "agreeing",
            Self::Indeterminate => "indeterminate",
            Self::Accuracy => "accuracy",
        }
    }
}

/// Code-switch detection row subtype.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SwitchMetricKind {
    /// Switched in both transcripts.
    Both,
    /// Switched in gold only.
    GoldOnly,
    /// Switched in main only.
    MainOnly,
    /// Switched in neither.
    Neither,
    /// Share of main's switches gold agrees with.
    Precision,
    /// Share of gold's switches main marked.
    Recall,
}

impl SwitchMetricKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Both => "both",
            Self::GoldOnly => "gold_only",
            Self::MainOnly => "main_only",
            Self::Neither => "neither",
            Self::Precision => "precision",
            Self::Recall => "recall",
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

/// One aligned word, with the evidence its scoring needs.
///
/// The token's status is set by the constructor that names the outcome, so a
/// token cannot say `Match` while its evidence describes a deletion. The gold
/// and main languages are side-typed, so they cannot be passed swapped.
pub(in crate::compare) struct ScoredWord<'a> {
    token: CompareToken,
    evidence: WordEvidence<'a>,
}

enum WordEvidence<'a> {
    Match {
        gold: &'a LanguagedWord<Gold>,
        main: &'a WordLanguage<Main>,
    },
    Insertion(InsertionAttribution<'a>),
    Deletion {
        gold: &'a LanguagedWord<Gold>,
    },
}

impl<'a> ScoredWord<'a> {
    pub(in crate::compare) fn matched(
        text: String,
        pos: Option<String>,
        gold: &'a LanguagedWord<Gold>,
        main: &'a WordLanguage<Main>,
    ) -> Self {
        Self {
            token: CompareToken {
                text,
                pos,
                status: CompareStatus::Match,
            },
            evidence: WordEvidence::Match { gold, main },
        }
    }

    pub(in crate::compare) fn inserted(
        text: String,
        pos: Option<String>,
        attribution: InsertionAttribution<'a>,
    ) -> Self {
        Self {
            token: CompareToken {
                text,
                pos,
                status: CompareStatus::ExtraMain,
            },
            evidence: WordEvidence::Insertion(attribution),
        }
    }

    pub(in crate::compare) fn deleted(
        text: String,
        pos: Option<String>,
        gold: &'a LanguagedWord<Gold>,
    ) -> Self {
        Self {
            token: CompareToken {
                text,
                pos,
                status: CompareStatus::ExtraGold,
            },
            evidence: WordEvidence::Deletion { gold },
        }
    }
}

/// Aggregate counters for one comparison.
///
/// Match, insertion and deletion totals are not counted here: they are derived
/// from the language tally in [`Self::finish`], so the legacy totals and the
/// per-language counts cannot disagree.
#[derive(Default)]
pub(in crate::compare) struct MetricAccumulator {
    displacement: DisplacementTally,
    pos_counts: BTreeMap<String, PosErrorCounts>,
    languages: LanguageTally,
}

impl MetricAccumulator {
    /// Count one aligned word and hand back its token for positioning.
    ///
    /// Words arrive in whole-file alignment order. Where a word counts for
    /// `cwer` is read off its own evidence, so no caller states it separately.
    /// No `PUNCT` check: `flatten_side` removes punctuation-tagged words before
    /// alignment, so none reaches here.
    pub(in crate::compare) fn record(&mut self, word: ScoredWord<'_>) -> CompareToken {
        let ScoredWord { token, evidence } = word;
        match evidence {
            WordEvidence::Match { gold, main } => {
                self.displacement.matched(gold.utterance());
                self.languages.matched(gold.language(), main);
            }
            WordEvidence::Insertion(attribution) => {
                match &attribution {
                    InsertionAttribution::InGold(site) => self
                        .displacement
                        .extra(site.utterances(), token.text.clone()),
                    InsertionAttribution::NoGoldWords => self.displacement.extra_without_gold(),
                }
                self.languages.inserted(&attribution);
            }
            WordEvidence::Deletion { gold } => {
                self.displacement
                    .missing(gold.utterance(), token.text.clone());
                self.languages.deleted(gold.language());
            }
        }
        let counts = self
            .pos_counts
            .entry(metric_pos_label(token.pos.as_deref()))
            .or_default();
        match token.status {
            CompareStatus::Match => counts.matches += 1,
            CompareStatus::ExtraMain => counts.insertions += 1,
            CompareStatus::ExtraGold => counts.deletions += 1,
        }
        token
    }

    /// A gold utterance mapped onto a main utterance.
    pub(in crate::compare) fn placed_utterance(
        &mut self,
        gold: &SidedLanguage<Gold>,
        main: &SidedLanguage<Main>,
    ) {
        self.languages.placed_utterance(gold, main);
    }

    /// A gold utterance with words that no main utterance matched.
    pub(in crate::compare) fn unplaced_utterance(&mut self, gold: &SidedLanguage<Gold>) {
        self.languages.unplaced_utterance(gold);
    }

    pub(in crate::compare) fn finish(self) -> CompareMetrics {
        CompareMetrics {
            displacement_edits: self.displacement.finish(),
            pos_counts: self.pos_counts,
            languages: self.languages.finish(),
        }
    }
}

/// The edit count behind `cwer`.
///
/// `cwer` is an order-insensitive error rate: a word the system recognised
/// correctly but placed in the wrong position cancels out instead of being
/// charged twice, once as a deletion and once as an insertion. Plain WER
/// conflates two failure modes that want separating, getting the word wrong
/// and getting the word right but placing it wrong, and for diarized ASR the
/// placement half is the merge pipeline's problem while the recognition half is
/// the engine's.
///
/// Cancellation is local to one gold utterance, not the whole file, so a word
/// that reappears in a distant utterance is still charged. A main word inside a
/// gold utterance cancels only there. A main word at the boundary between two
/// gold utterances may cancel in either, because nothing in the alignment says
/// which one it drifted from: a word moved to the end of its utterance and a
/// word moved to the start of the next one land in the same place. Matching
/// uses the aligner's case-insensitive rule, so the two cannot disagree about
/// what counts as the same word.
///
/// Words arrive in alignment order, which visits gold utterances in order, so
/// one utterance is open at a time. When the next opens, the open one settles:
/// its own main words cancel against its missing gold words first, since they
/// can cancel nowhere else; the boundary words after it take what remains; the
/// boundary words left over are carried into the next utterance. That order
/// cancels as many words as any assignment could.
///
/// The name `cwer` and the cancellation idea are taken from an upstream fork of
/// this repository, which introduced the metric first.
#[derive(Default)]
struct DisplacementTally {
    /// Edits settled so far.
    edits: usize,
    open: Option<OpenUtterance>,
}

/// The gold utterance the alignment is in, with its unsettled words.
struct OpenUtterance {
    utterance: UtteranceIndex<Gold>,
    /// Its gold words the main transcript lacks.
    missing: Vec<String>,
    /// Main words that can cancel only here: inside the utterance, or carried
    /// from the boundary before it.
    extra: Vec<String>,
    /// Main words at the boundary after it, each with the utterance that
    /// follows, where it may cancel instead.
    boundary: Vec<(UtteranceIndex<Gold>, String)>,
}

impl DisplacementTally {
    fn matched(&mut self, utterance: UtteranceIndex<Gold>) {
        self.enter(utterance);
    }

    fn missing(&mut self, utterance: UtteranceIndex<Gold>, word: String) {
        self.enter(utterance).missing.push(word);
    }

    fn extra(&mut self, site: SiteUtterances, word: String) {
        match site {
            SiteUtterances::Within(utterance) => self.enter(utterance).extra.push(word),
            SiteUtterances::Across { before, after } => {
                self.enter(before).boundary.push((after, word));
            }
        }
    }

    /// A main word with no gold words to cancel against: an edit now.
    fn extra_without_gold(&mut self) {
        self.edits += 1;
    }

    /// Open `utterance`, settling the open one first when it is another.
    fn enter(&mut self, utterance: UtteranceIndex<Gold>) -> &mut OpenUtterance {
        let carried = match self.open.take() {
            Some(open) if open.utterance == utterance => return self.open.insert(open),
            Some(open) => self.settle(open),
            None => Vec::new(),
        };
        let mut extra = Vec::with_capacity(carried.len());
        for (target, word) in carried {
            match target == utterance {
                true => extra.push(word),
                // Carried toward an utterance the alignment did not enter
                // next: nothing is left to cancel against.
                false => self.edits += 1,
            }
        }
        self.open.insert(OpenUtterance {
            utterance,
            missing: Vec::new(),
            extra,
            boundary: Vec::new(),
        })
    }

    /// Settle one utterance, returning the boundary words it could not cancel.
    fn settle(&mut self, open: OpenUtterance) -> Vec<(UtteranceIndex<Gold>, String)> {
        let OpenUtterance {
            mut missing,
            extra,
            boundary,
            ..
        } = open;
        for word in extra {
            match take_partner(&mut missing, &word) {
                Some(_) => {}
                None => self.edits += 1,
            }
        }
        let carried = boundary
            .into_iter()
            .filter(|(_, word)| take_partner(&mut missing, word).is_none())
            .collect();
        self.edits += missing.len();
        carried
    }

    fn finish(mut self) -> usize {
        let carried = match self.open.take() {
            Some(open) => self.settle(open),
            None => Vec::new(),
        };
        self.edits + carried.len()
    }
}

/// Remove and return a missing gold word equal to `word`, if there is one.
///
/// `swap_remove`: cancellation is a set operation, so which surviving entry
/// moves is irrelevant and the order of `missing` is never observed.
fn take_partner(missing: &mut Vec<String>, word: &str) -> Option<String> {
    missing
        .iter()
        .position(|gold| gold.eq_ignore_ascii_case(word))
        .map(|position| missing.swap_remove(position))
}

pub(in crate::compare) fn metric_pos_label(pos: Option<&str>) -> String {
    pos.unwrap_or("?").to_uppercase()
}

/// Serialize comparison metrics as CSV rows with header.
pub fn format_metrics_csv(metrics: &CompareMetrics) -> Result<String, CompareSerializationError> {
    CompareMetricsCsvTable::from_metrics(metrics)?.to_csv_string()
}

#[cfg(test)]
mod tests {
    use super::super::language::languaged_utterances;
    use super::super::tests::{chat_text, parse_lenient};
    use super::*;
    use talkbank_parser::TreeSitterParser;

    /// Settling cancels an utterance's own main words before its boundary
    /// words, which is what makes the count the fewest edits any assignment of
    /// words to utterances could give.
    ///
    /// Utterance 0 lacks one `w` and has an extra `w` inside it; a boundary `w`
    /// follows it; utterance 1 lacks a `w`. The inner `w` can cancel only in
    /// utterance 0 and the boundary `w` in either, so both cancel: 0 edits.
    /// Letting the boundary `w` take utterance 0's missing word first strands
    /// both the inner `w` and utterance 1's missing one: 2 edits.
    #[test]
    fn settling_cancels_an_utterances_own_words_before_boundary_words() {
        let parser = TreeSitterParser::new().expect("parser");
        let (file, _) = parse_lenient(
            &parser,
            &chat_text("eng", &[("CHI", "a w b ."), ("CHI", "w c .")]),
        );
        let utterances = languaged_utterances::<Gold>(&file);
        let mut indices = utterances
            .iter()
            .map(|utterance| utterance.utterance_index());
        let (first, second) = (
            indices.next().expect("first utterance"),
            indices.next().expect("second utterance"),
        );

        let mut tally = DisplacementTally::default();
        tally.missing(first, "w".to_string());
        tally.extra(SiteUtterances::Within(first), "w".to_string());
        tally.extra(
            SiteUtterances::Across {
                before: first,
                after: second,
            },
            "w".to_string(),
        );
        tally.missing(second, "w".to_string());

        assert_eq!(tally.finish(), 0);
    }
}
