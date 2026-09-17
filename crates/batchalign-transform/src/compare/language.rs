//! The language compare scores each word and utterance in, and the tallies
//! built from it.
//!
//! # The graph
//!
//! ```text
//! ChatFile
//!   -> languaged_utterances::<Gold | Main>   one walk: each utterance's words and its language
//!   -> LanguagedWord<Side>                   a word that knows its language and its utterance
//!   -> alignment (engine)                    match / insertion at an InsertionSite / deletion
//!   -> ScoredWord (metrics)                  the outcome and its side-typed evidence
//!   -> MetricAccumulator::record             counts it, cancelling `cwer` by gold utterance
//!   -> LanguageTally::finish                 LanguageScores
//! ```
//!
//! Every language value is marked with the document it came from ([`Gold`],
//! [`Main`], or [`Run`] for a cross-run comparison, where neither side is a
//! reference). A tally that takes a gold language and a main language takes
//! `SidedLanguage<Gold>` and `SidedLanguage<Main>`, so passing them in the wrong
//! order does not compile.
//!
//! # Where a word's language comes from
//!
//! A word's language is governed by the nearest mark: its own `@s`, else an
//! enclosing `<...> [@s]` span, else its utterance. Every word, marked or not,
//! is resolved through chatter's [`ExtractedWord::resolve_language`], so the
//! word-level rule is never re-derived here. The utterance's own language (a
//! `[- lang]` precode, else the first `@Languages` entry) IS derived here, as
//! the tier language the resolver needs: chatter computes the same baseline
//! only as a side effect of validation and exposes no function for it, and
//! Batchalign's morphotag payload writes its own copy with a job-level fallback.
//! One shared chatter function is owed.
//!
//! # Agreement is three-valued
//!
//! Two languages agree, differ, or cannot be compared ([`LanguageAgreement`]).
//! An unresolved or ambiguous language is never "the same" as another: two
//! transcripts that both failed to say what language an utterance is in have
//! not agreed about anything, and counting that as agreement would inflate
//! accuracy exactly where the evidence is weakest.
//!
//! # Substitutions and attribution
//!
//! Between two consecutive matches of the whole-file alignment, a deletion and
//! an insertion are one wrong word, so up to `min(deletions, insertions)` of
//! them pair into substitutions, charged to the gold word's language. Pairing
//! never crosses a match. This is the edit count for the aligner's own matches;
//! a pure Levenshtein alignment can trade a match for substitutions and report
//! fewer edits.
//!
//! A gold word's errors belong to its language. An unpaired insertion belongs
//! to the language of the gold word that owns its [`InsertionSite`]: the next
//! gold word in the alignment, or the last when none follows. Only an insertion
//! against a gold transcript with no compared words at all is unattributed. The
//! main transcript's own language is never used: in code-switched speech it is
//! exactly the evidence under test. Main utterance boundaries play no part, so a
//! main transcript split differently scores the same.

use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::marker::PhantomData;

use talkbank_model::SpeakerCode;
use talkbank_model::alignment::helpers::PositionalDomain;
use talkbank_model::model::{ChatFile, LanguageCode};
use talkbank_model::validation::{GoverningMarkKind, LanguageResolution};

use crate::extract::{self, ExtractedWord};

use super::metrics::{
    AgreementMetricKind, AgreementScope, CompareMetricName, CompareMetricValue,
    CompareMetricsCsvRow, LanguageMetricKind, SwitchMetricKind,
};

/// The document a language value came from. Sealed to this module's markers.
pub trait Side: sealed::Sealed + Copy + Eq + std::fmt::Debug {}

mod sealed {
    pub trait Sealed {}
}

/// The reference transcript of a comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gold {}

/// The transcript being scored against the reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Main {}

/// Either transcript of a cross-run comparison, where neither is a reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Run {}

impl sealed::Sealed for Gold {}
impl sealed::Sealed for Main {}
impl sealed::Sealed for Run {}
impl Side for Gold {}
impl Side for Main {}
impl Side for Run {}

/// A position in one side's list of utterances.
///
/// Side-marked, so a main utterance cannot stand where a gold one is meant, and
/// minted only by [`languaged_utterances`]: no other code builds one from a bare
/// number.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::compare) struct UtteranceIndex<S: Side> {
    raw: usize,
    side: PhantomData<S>,
}

impl<S: Side> UtteranceIndex<S> {
    /// The position, for the public output records that carry a bare index.
    pub(in crate::compare) fn raw(self) -> usize {
        self.raw
    }
}

/// One value per utterance of one side, indexed only by that side's
/// [`UtteranceIndex`].
///
/// The root collection is the utterances themselves, built by
/// [`languaged_utterances`] alongside the indices; every other one is derived
/// from it with [`Self::map_ref`]. So each has exactly one value per utterance,
/// and no bare number reaches one.
#[derive(Debug, Clone)]
pub(in crate::compare) struct PerUtterance<S: Side, T> {
    values: Vec<T>,
    side: PhantomData<S>,
}

impl<S: Side, T> PerUtterance<S, T> {
    /// A value for each utterance, computed from this collection's value for it.
    pub(in crate::compare) fn map_ref<U>(&self, value: impl FnMut(&T) -> U) -> PerUtterance<S, U> {
        PerUtterance {
            values: self.values.iter().map(value).collect(),
            side: PhantomData,
        }
    }

    /// Every value, in utterance order.
    pub(in crate::compare) fn iter(&self) -> std::slice::Iter<'_, T> {
        self.values.iter()
    }

    /// Every value, mutably, in utterance order.
    pub(in crate::compare) fn values_mut(&mut self) -> std::slice::IterMut<'_, T> {
        self.values.iter_mut()
    }
}

impl<'s, S: Side, T> IntoIterator for &'s PerUtterance<S, T> {
    type Item = &'s T;
    type IntoIter = std::slice::Iter<'s, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.values.iter()
    }
}

impl<S: Side, T> std::ops::Index<UtteranceIndex<S>> for PerUtterance<S, T> {
    type Output = T;

    fn index(&self, utterance: UtteranceIndex<S>) -> &T {
        &self.values[utterance.raw]
    }
}

impl<S: Side, T> std::ops::IndexMut<UtteranceIndex<S>> for PerUtterance<S, T> {
    fn index_mut(&mut self, utterance: UtteranceIndex<S>) -> &mut T {
        &mut self.values[utterance.raw]
    }
}

/// The language a comparison attributes a word or utterance to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LanguageBucket {
    /// One definite language.
    Language(LanguageCode),
    /// A word marked as mixing languages, such as `@s:eng+spa`, with its codes.
    Mixed(Vec<LanguageCode>),
    /// A word marked as ambiguous between languages, such as `@s:eng&spa`,
    /// with its codes.
    Ambiguous(Vec<LanguageCode>),
    /// No language could be resolved: an utterance with no precode in a file
    /// with no `@Languages`, or a word mark the resolver could not resolve.
    Unresolved,
}

impl LanguageBucket {
    fn of_resolution(resolution: LanguageResolution) -> Self {
        match resolution {
            LanguageResolution::Single(code) => Self::Language(code),
            LanguageResolution::Multiple(codes) => Self::Mixed(codes),
            LanguageResolution::Ambiguous(codes) => Self::Ambiguous(codes),
            LanguageResolution::Unresolved => Self::Unresolved,
        }
    }

    /// The key segment used in metrics CSV rows, in CHAT's own spelling:
    /// `eng`, `eng+spa`, `eng&spa`, or `unresolved`.
    pub fn csv_key(&self) -> String {
        match self {
            Self::Language(code) => code.as_str().to_string(),
            Self::Mixed(codes) => join_codes(codes, "+"),
            Self::Ambiguous(codes) => join_codes(codes, "&"),
            Self::Unresolved => "unresolved".to_string(),
        }
    }

    /// Variant order for [`Ord`]: single languages, mixes, ambiguities, then
    /// unresolved.
    fn rank(&self) -> u8 {
        match self {
            Self::Language(_) => 0,
            Self::Mixed(_) => 1,
            Self::Ambiguous(_) => 2,
            Self::Unresolved => 3,
        }
    }
}

fn join_codes(codes: &[LanguageCode], separator: &str) -> String {
    codes
        .iter()
        .map(LanguageCode::as_str)
        .collect::<Vec<_>>()
        .join(separator)
}

/// A total order so tallies iterate, and CSV rows print, deterministically.
///
/// Written by hand because [`LanguageCode`] has no order of its own. It agrees
/// with `Eq`: same variant, then the codes compared as strings in order.
impl Ord for LanguageBucket {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Self::Language(left), Self::Language(right)) => left.as_str().cmp(right.as_str()),
            (Self::Mixed(left), Self::Mixed(right))
            | (Self::Ambiguous(left), Self::Ambiguous(right)) => left
                .iter()
                .map(LanguageCode::as_str)
                .cmp(right.iter().map(LanguageCode::as_str)),
            _ => self.rank().cmp(&other.rank()),
        }
    }
}

impl PartialOrd for LanguageBucket {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Whether two languages are the same, differ, or cannot be compared.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::compare) enum LanguageAgreement {
    /// The same definite language, or the same mix.
    Same,
    /// Two definite languages or mixes that differ.
    Different,
    /// At least one side is unresolved or ambiguous.
    Indeterminate,
}

impl LanguageAgreement {
    fn of(left: &LanguageBucket, right: &LanguageBucket) -> Self {
        use LanguageBucket::{Ambiguous, Language, Mixed, Unresolved};
        match (left, right) {
            (Unresolved, _) | (_, Unresolved) | (Ambiguous(_), _) | (_, Ambiguous(_)) => {
                Self::Indeterminate
            }
            (Language(_), Language(_)) | (Mixed(_), Mixed(_)) => match left == right {
                true => Self::Same,
                false => Self::Different,
            },
            (Language(_), Mixed(_)) | (Mixed(_), Language(_)) => Self::Different,
        }
    }
}

/// A language, marked with the document it came from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::compare) struct SidedLanguage<S: Side> {
    bucket: LanguageBucket,
    side: PhantomData<S>,
}

impl<S: Side> SidedLanguage<S> {
    fn new(bucket: LanguageBucket) -> Self {
        Self {
            bucket,
            side: PhantomData,
        }
    }
}

/// Whether a word is in its utterance's language or switched out of it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::compare) enum LanguageRole {
    /// Unmarked, or marked with the utterance's own language.
    UtteranceLanguage,
    /// Marked, by its own `@s` or a span, as a language other than the
    /// utterance's, including a mark whose language could not be resolved:
    /// the transcriber still marked the word as a switch.
    Switched,
}

/// One word's language and whether it is a switch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::compare) struct WordLanguage<S: Side> {
    language: SidedLanguage<S>,
    role: LanguageRole,
}

/// A word that already knows its language and the utterance it is in. Built
/// only by [`languaged_utterances`], which reads both from the same line.
#[derive(Debug, Clone)]
pub(in crate::compare) struct LanguagedWord<S: Side> {
    word: ExtractedWord,
    utterance: UtteranceIndex<S>,
    language: WordLanguage<S>,
}

impl<S: Side> LanguagedWord<S> {
    pub(in crate::compare) fn extracted(&self) -> &ExtractedWord {
        &self.word
    }

    pub(in crate::compare) fn utterance(&self) -> UtteranceIndex<S> {
        self.utterance
    }

    pub(in crate::compare) fn language(&self) -> &WordLanguage<S> {
        &self.language
    }
}

/// One utterance's compare-domain words, each with its language, and the
/// utterance's own language. Built only by [`languaged_utterances`].
#[derive(Debug, Clone)]
pub(in crate::compare) struct LanguagedUtterance<S: Side> {
    speaker: SpeakerCode,
    utterance_index: UtteranceIndex<S>,
    language: SidedLanguage<S>,
    words: Vec<LanguagedWord<S>>,
}

impl<S: Side> LanguagedUtterance<S> {
    pub(in crate::compare) fn speaker(&self) -> &SpeakerCode {
        &self.speaker
    }

    pub(in crate::compare) fn utterance_index(&self) -> UtteranceIndex<S> {
        self.utterance_index
    }

    pub(in crate::compare) fn language(&self) -> &SidedLanguage<S> {
        &self.language
    }

    pub(in crate::compare) fn words(&self) -> &[LanguagedWord<S>] {
        &self.words
    }
}

/// Every utterance of one document, words and languages from one walk.
///
/// A word's language comes from the very utterance line its text does.
/// Extracting words and reading precodes in two walks and zipping them would
/// pair them by position and nothing else.
pub(in crate::compare) fn languaged_utterances<S: Side>(
    chat_file: &ChatFile,
) -> PerUtterance<S, LanguagedUtterance<S>> {
    let declared: &[LanguageCode] = &chat_file.languages;
    let primary = declared.first();

    let values = chat_file
        .utterances()
        .enumerate()
        .map(|(index, utterance)| {
            let utterance_index = UtteranceIndex {
                raw: index,
                side: PhantomData,
            };
            let tier_language = utterance.main.content.language_code.as_ref().or(primary);
            let utterance_bucket = match tier_language {
                Some(code) => LanguageBucket::Language(code.clone()),
                None => LanguageBucket::Unresolved,
            };

            let mut extracted = Vec::new();
            extract::collect_utterance_content(
                &utterance.main.content.content,
                PositionalDomain::Mor,
                &mut extracted,
            );
            let words = extracted
                .into_iter()
                .map(|word| {
                    let language =
                        resolve_word_language(&word, &utterance_bucket, tier_language, declared);
                    LanguagedWord {
                        word,
                        utterance: utterance_index,
                        language,
                    }
                })
                .collect();

            LanguagedUtterance {
                speaker: SpeakerCode::new(&utterance.main.speaker),
                utterance_index,
                language: SidedLanguage::new(utterance_bucket),
                words,
            }
        })
        .collect();

    PerUtterance {
        values,
        side: PhantomData,
    }
}

/// Resolve one word's language through chatter's governing-mark resolver, and
/// classify it as a switch by its mark.
///
/// The resolver's diagnostics are not kept: a mark it cannot resolve already
/// shows up as `Unresolved` in the scores, which is the consequence a
/// comparison reports.
fn resolve_word_language<S: Side>(
    word: &ExtractedWord,
    utterance_language: &LanguageBucket,
    tier_language: Option<&LanguageCode>,
    declared: &[LanguageCode],
) -> WordLanguage<S> {
    let bucket =
        LanguageBucket::of_resolution(word.resolve_language(tier_language, declared).resolution);
    let role = match (
        word.language_kind(),
        LanguageAgreement::of(&bucket, utterance_language),
    ) {
        (GoverningMarkKind::Utterance, _)
        | (GoverningMarkKind::Own | GoverningMarkKind::Span, LanguageAgreement::Same) => {
            LanguageRole::UtteranceLanguage
        }
        (
            GoverningMarkKind::Own | GoverningMarkKind::Span,
            LanguageAgreement::Different | LanguageAgreement::Indeterminate,
        ) => LanguageRole::Switched,
    };
    WordLanguage {
        language: SidedLanguage::new(bucket),
        role,
    }
}

/// A rate, or the fact that there was nothing to divide by.
///
/// Not a `0.0` for an empty denominator: a switch precision of zero means every
/// predicted switch was wrong, which is a different finding from there being no
/// predicted switches at all.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Rate {
    /// The ratio.
    Defined(f64),
    /// The denominator was zero.
    NoDenominator,
}

impl Rate {
    pub(in crate::compare) fn of(numerator: usize, denominator: usize) -> Self {
        match denominator {
            0 => Self::NoDenominator,
            denominator => Self::Defined(numerator as f64 / denominator as f64),
        }
    }
}

/// Error counts for the gold words of one language.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LanguageErrorCounts {
    matches: usize,
    substitutions: usize,
    deletions: usize,
    insertions: usize,
}

impl LanguageErrorCounts {
    /// Gold words matched by a main word.
    pub fn matches(&self) -> usize {
        self.matches
    }

    /// Gold words paired with a wrong main word.
    pub fn substitutions(&self) -> usize {
        self.substitutions
    }

    /// Gold words with no main counterpart.
    pub fn deletions(&self) -> usize {
        self.deletions
    }

    /// Unpaired main words attributed to this language.
    pub fn insertions(&self) -> usize {
        self.insertions
    }

    /// Gold words in this language.
    pub fn gold_words(&self) -> usize {
        self.matches + self.substitutions + self.deletions
    }

    /// `(substitutions + deletions + insertions) / gold words`.
    pub fn wer_with_substitutions(&self) -> Rate {
        Rate::of(
            self.substitutions + self.deletions + self.insertions,
            self.gold_words(),
        )
    }

    fn write_csv_rows(&self, language: &LanguageBucket, rows: &mut Vec<CompareMetricsCsvRow>) {
        // Destructured without `..`: a new count must be given a row here or
        // this stops compiling.
        let Self {
            matches,
            substitutions,
            deletions,
            insertions,
        } = self;
        for (kind, value) in [
            (
                LanguageMetricKind::GoldWords,
                CompareMetricValue::Count(self.gold_words()),
            ),
            (
                LanguageMetricKind::Matches,
                CompareMetricValue::Count(*matches),
            ),
            (
                LanguageMetricKind::Substitutions,
                CompareMetricValue::Count(*substitutions),
            ),
            (
                LanguageMetricKind::Deletions,
                CompareMetricValue::Count(*deletions),
            ),
            (
                LanguageMetricKind::Insertions,
                CompareMetricValue::Count(*insertions),
            ),
            (
                LanguageMetricKind::WerWithSubstitutions,
                CompareMetricValue::Rate(self.wer_with_substitutions()),
            ),
        ] {
            rows.push(CompareMetricsCsvRow::new(
                CompareMetricName::Language {
                    language: language.clone(),
                    kind,
                },
                value,
            ));
        }
    }
}

/// A gold language and a main language, the key of a confusion count.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct LanguagePair {
    /// The language the gold transcript gives.
    pub gold: LanguageBucket,
    /// The language the main transcript gives.
    pub main: LanguageBucket,
}

/// How often each gold language was given each main language.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LanguageConfusion {
    counts: BTreeMap<LanguagePair, usize>,
}

impl LanguageConfusion {
    fn record(&mut self, gold: &SidedLanguage<Gold>, main: &SidedLanguage<Main>) {
        *self
            .counts
            .entry(LanguagePair {
                gold: gold.bucket.clone(),
                main: main.bucket.clone(),
            })
            .or_default() += 1;
    }

    /// Every observed pair with its count, in order.
    pub fn pairs(&self) -> impl Iterator<Item = (&LanguagePair, usize)> {
        self.counts.iter().map(|(pair, count)| (pair, *count))
    }

    /// Items scored, whatever their agreement.
    pub fn scored(&self) -> usize {
        self.counts.values().sum()
    }

    fn with_agreement(&self, agreement: LanguageAgreement) -> usize {
        self.pairs()
            .filter(|(pair, _)| LanguageAgreement::of(&pair.gold, &pair.main) == agreement)
            .map(|(_, count)| count)
            .sum()
    }

    /// Items whose two languages are the same.
    pub fn agreeing(&self) -> usize {
        self.with_agreement(LanguageAgreement::Same)
    }

    /// Items where either language is unresolved or ambiguous.
    pub fn indeterminate(&self) -> usize {
        self.with_agreement(LanguageAgreement::Indeterminate)
    }

    /// `agreeing / (agreeing + differing)`: indeterminate items are neither
    /// right nor wrong, so they are left out of the denominator.
    pub fn accuracy(&self) -> Rate {
        let agreeing = self.agreeing();
        Rate::of(
            agreeing,
            agreeing + self.with_agreement(LanguageAgreement::Different),
        )
    }

    fn write_csv_rows(&self, scope: AgreementScope, rows: &mut Vec<CompareMetricsCsvRow>) {
        for (kind, value) in [
            (
                AgreementMetricKind::Scored,
                CompareMetricValue::Count(self.scored()),
            ),
            (
                AgreementMetricKind::Agreeing,
                CompareMetricValue::Count(self.agreeing()),
            ),
            (
                AgreementMetricKind::Indeterminate,
                CompareMetricValue::Count(self.indeterminate()),
            ),
            (
                AgreementMetricKind::Accuracy,
                CompareMetricValue::Rate(self.accuracy()),
            ),
        ] {
            rows.push(CompareMetricsCsvRow::new(
                CompareMetricName::Agreement { scope, kind },
                value,
            ));
        }
        for (pair, count) in self.pairs() {
            rows.push(CompareMetricsCsvRow::new(
                CompareMetricName::Confusion {
                    scope,
                    gold: pair.gold.clone(),
                    main: pair.main.clone(),
                },
                CompareMetricValue::Count(count),
            ));
        }
    }
}

/// Utterance language agreement, for gold utterances compare could place.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct UtteranceLanguageAgreement {
    placed: LanguageConfusion,
    unplaced: BTreeMap<LanguageBucket, usize>,
}

impl UtteranceLanguageAgreement {
    /// Gold utterances mapped to a main utterance, by both languages.
    pub fn placed(&self) -> &LanguageConfusion {
        &self.placed
    }

    /// Gold utterances with words that no main utterance matched, by gold
    /// language. Not disagreements: there was nothing to compare against.
    pub fn unplaced(&self) -> &BTreeMap<LanguageBucket, usize> {
        &self.unplaced
    }

    fn write_csv_rows(&self, rows: &mut Vec<CompareMetricsCsvRow>) {
        let Self { placed, unplaced } = self;
        placed.write_csv_rows(AgreementScope::Utterance, rows);
        for (language, count) in unplaced {
            rows.push(CompareMetricsCsvRow::new(
                CompareMetricName::UtteranceLanguageUnplaced {
                    language: language.clone(),
                },
                CompareMetricValue::Count(*count),
            ));
        }
    }
}

/// Code-switch detection over matched words.
///
/// Scored on matched pairs only, so it measures whether switches were marked
/// on words that were recognized. Words that were not recognized are the
/// per-language error rates' business.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SwitchAgreement {
    both_switched: usize,
    gold_only: usize,
    main_only: usize,
    neither: usize,
}

impl SwitchAgreement {
    fn record(&mut self, gold: &WordLanguage<Gold>, main: &WordLanguage<Main>) {
        let cell = match (gold.role, main.role) {
            (LanguageRole::Switched, LanguageRole::Switched) => &mut self.both_switched,
            (LanguageRole::Switched, LanguageRole::UtteranceLanguage) => &mut self.gold_only,
            (LanguageRole::UtteranceLanguage, LanguageRole::Switched) => &mut self.main_only,
            (LanguageRole::UtteranceLanguage, LanguageRole::UtteranceLanguage) => &mut self.neither,
        };
        *cell += 1;
    }

    /// Matched words switched in both transcripts.
    pub fn both_switched(&self) -> usize {
        self.both_switched
    }

    /// Matched words switched in gold only: missed switches.
    pub fn gold_only(&self) -> usize {
        self.gold_only
    }

    /// Matched words switched in main only: spurious switches.
    pub fn main_only(&self) -> usize {
        self.main_only
    }

    /// Matched words switched in neither.
    pub fn neither(&self) -> usize {
        self.neither
    }

    /// Of the switches main marked, the share gold agrees with.
    pub fn precision(&self) -> Rate {
        Rate::of(self.both_switched, self.both_switched + self.main_only)
    }

    /// Of the switches gold marks, the share main marked too.
    pub fn recall(&self) -> Rate {
        Rate::of(self.both_switched, self.both_switched + self.gold_only)
    }

    fn write_csv_rows(&self, rows: &mut Vec<CompareMetricsCsvRow>) {
        let Self {
            both_switched,
            gold_only,
            main_only,
            neither,
        } = self;
        for (kind, value) in [
            (
                SwitchMetricKind::Both,
                CompareMetricValue::Count(*both_switched),
            ),
            (
                SwitchMetricKind::GoldOnly,
                CompareMetricValue::Count(*gold_only),
            ),
            (
                SwitchMetricKind::MainOnly,
                CompareMetricValue::Count(*main_only),
            ),
            (
                SwitchMetricKind::Neither,
                CompareMetricValue::Count(*neither),
            ),
            (
                SwitchMetricKind::Precision,
                CompareMetricValue::Rate(self.precision()),
            ),
            (
                SwitchMetricKind::Recall,
                CompareMetricValue::Rate(self.recall()),
            ),
        ] {
            rows.push(CompareMetricsCsvRow::new(
                CompareMetricName::Switch(kind),
                value,
            ));
        }
    }
}

/// Every language score of one comparison, and the one source of the
/// match, insertion and deletion totals.
///
/// Nonempty scores come only from [`LanguageTally::finish`]. `Default` is the
/// scores of a comparison that observed nothing.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LanguageScores {
    by_language: BTreeMap<LanguageBucket, LanguageErrorCounts>,
    unattributed_insertions: usize,
    utterances: UtteranceLanguageAgreement,
    words: LanguageConfusion,
    switches: SwitchAgreement,
}

impl LanguageScores {
    /// Error counts for every gold language, in order.
    pub fn by_language(&self) -> impl Iterator<Item = (&LanguageBucket, &LanguageErrorCounts)> {
        self.by_language.iter()
    }

    /// Unpaired insertions with no gold word to attribute them to, charged to
    /// no language: every insertion against a gold with no compared words.
    pub fn unattributed_insertions(&self) -> usize {
        self.unattributed_insertions
    }

    /// Utterance language agreement.
    pub fn utterances(&self) -> &UtteranceLanguageAgreement {
        &self.utterances
    }

    /// Word language agreement over matched words.
    pub fn words(&self) -> &LanguageConfusion {
        &self.words
    }

    /// Code-switch detection over matched words.
    pub fn switches(&self) -> &SwitchAgreement {
        &self.switches
    }

    fn sum(&self, count: impl Fn(&LanguageErrorCounts) -> usize) -> usize {
        self.by_language.values().map(count).sum()
    }

    /// Gold words matched by a main word.
    pub fn matches(&self) -> usize {
        self.sum(LanguageErrorCounts::matches)
    }

    /// Substitutions across every language.
    pub fn substitutions(&self) -> usize {
        self.sum(LanguageErrorCounts::substitutions)
    }

    /// Gold words not matched: substituted or deleted.
    pub fn unmatched_gold_words(&self) -> usize {
        self.sum(|counts| counts.substitutions + counts.deletions)
    }

    /// Main words not matched: substituting, inserted, or unattributed.
    pub fn unmatched_main_words(&self) -> usize {
        self.sum(|counts| counts.substitutions + counts.insertions) + self.unattributed_insertions
    }

    /// Gold words in every language.
    pub fn gold_words(&self) -> usize {
        self.sum(LanguageErrorCounts::gold_words)
    }

    /// `(substitutions + deletions + insertions) / gold words`, over every
    /// language plus the unattributed insertions.
    pub fn wer_with_substitutions(&self) -> Rate {
        let edits = self.sum(|counts| counts.substitutions + counts.deletions + counts.insertions)
            + self.unattributed_insertions;
        Rate::of(edits, self.gold_words())
    }

    /// Append every language row, in a fixed order.
    ///
    /// Every count is written, not only rates, so a corpus roll-up sums rows
    /// and recomputes rates rather than averaging per-file ratios.
    pub(in crate::compare) fn write_csv_rows(&self, rows: &mut Vec<CompareMetricsCsvRow>) {
        let Self {
            by_language,
            unattributed_insertions,
            utterances,
            words,
            switches,
        } = self;
        rows.push(CompareMetricsCsvRow::new(
            CompareMetricName::Substitutions,
            CompareMetricValue::Count(self.substitutions()),
        ));
        rows.push(CompareMetricsCsvRow::new(
            CompareMetricName::WerWithSubstitutions,
            CompareMetricValue::Rate(self.wer_with_substitutions()),
        ));
        rows.push(CompareMetricsCsvRow::new(
            CompareMetricName::UnattributedInsertions,
            CompareMetricValue::Count(*unattributed_insertions),
        ));
        for (language, counts) in by_language {
            counts.write_csv_rows(language, rows);
        }
        utterances.write_csv_rows(rows);
        words.write_csv_rows(AgreementScope::Word, rows);
        switches.write_csv_rows(rows);
    }
}

/// A gold word, however a stage holds it: what an [`InsertionSite`] needs to
/// know about its neighbours.
pub(in crate::compare) trait GoldNeighbour: Copy {
    /// The gold utterance the word is in.
    fn gold_utterance(self) -> UtteranceIndex<Gold>;
}

impl GoldNeighbour for &LanguagedWord<Gold> {
    fn gold_utterance(self) -> UtteranceIndex<Gold> {
        self.utterance
    }
}

/// Where a main word the gold lacks sits among the gold words, in alignment
/// order.
///
/// Built once per insertion, from the gold tokens the alignment has consumed.
/// Everything that depends on an insertion's position reads it here, so none
/// of them can disagree: the gold word that owns its error ([`Self::owner`]),
/// where the gold view shows it, and which gold utterances its `cwer`
/// cancellation may use ([`Self::utterances`]). It holds neighbours only;
/// whether they share an utterance is derived, never stored, so no site can
/// claim a relation its words do not have.
#[derive(Debug, Clone, Copy)]
pub(in crate::compare) enum InsertionSite<W> {
    /// Before the first gold word.
    BeforeGold { next: W },
    /// Between two consecutive gold words.
    BetweenGoldWords { previous: W, next: W },
    /// After the last gold word.
    AfterGold { previous: W },
}

/// The gold utterances an [`InsertionSite`] touches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::compare) enum SiteUtterances {
    /// Inside one utterance, or at the start or end of the gold.
    Within(UtteranceIndex<Gold>),
    /// At the boundary between two utterances.
    Across {
        before: UtteranceIndex<Gold>,
        after: UtteranceIndex<Gold>,
    },
}

impl<W: GoldNeighbour> InsertionSite<W> {
    /// The gold word the insertion's error belongs to: the next gold word, or
    /// the last when none follows.
    pub(in crate::compare) fn owner(self) -> W {
        match self {
            Self::BeforeGold { next } | Self::BetweenGoldWords { next, .. } => next,
            Self::AfterGold { previous } => previous,
        }
    }

    /// The gold utterances the site touches.
    pub(in crate::compare) fn utterances(self) -> SiteUtterances {
        match self {
            Self::BeforeGold { next } => SiteUtterances::Within(next.gold_utterance()),
            Self::AfterGold { previous } => SiteUtterances::Within(previous.gold_utterance()),
            Self::BetweenGoldWords { previous, next } => {
                let (before, after) = (previous.gold_utterance(), next.gold_utterance());
                match before == after {
                    true => SiteUtterances::Within(before),
                    false => SiteUtterances::Across { before, after },
                }
            }
        }
    }

    /// The same site over another view of the same gold words.
    pub(in crate::compare) fn map<V>(self, view: impl Fn(W) -> V) -> InsertionSite<V> {
        match self {
            Self::BeforeGold { next } => InsertionSite::BeforeGold { next: view(next) },
            Self::BetweenGoldWords { previous, next } => InsertionSite::BetweenGoldWords {
                previous: view(previous),
                next: view(next),
            },
            Self::AfterGold { previous } => InsertionSite::AfterGold {
                previous: view(previous),
            },
        }
    }
}

/// Where an unpaired main word's error belongs.
pub(in crate::compare) enum InsertionAttribution<'a> {
    /// Among the gold words: to the site's owner.
    InGold(InsertionSite<&'a LanguagedWord<Gold>>),
    /// The gold has no compared words, so there is no gold language to charge.
    NoGoldWords,
}

/// An insertion awaiting pairing: [`InsertionAttribution`] with the gold
/// language copied out, so the gap owns what it holds.
#[derive(Debug)]
enum PendingInsertion {
    Attributed(LanguageBucket),
    Unattributed,
}

/// The errors seen since the last match, awaiting pairing.
#[derive(Debug, Default)]
struct OpenGap {
    deletions: Vec<LanguageBucket>,
    insertions: Vec<PendingInsertion>,
}

/// Accumulates language scores during one comparison.
#[derive(Debug, Default)]
pub(in crate::compare) struct LanguageTally {
    scores: LanguageScores,
    gap: OpenGap,
}

impl LanguageTally {
    pub(in crate::compare) fn matched(
        &mut self,
        gold: &WordLanguage<Gold>,
        main: &WordLanguage<Main>,
    ) {
        self.close_gap();
        self.counts(&gold.language.bucket).matches += 1;
        self.scores.words.record(&gold.language, &main.language);
        self.scores.switches.record(gold, main);
    }

    pub(in crate::compare) fn deleted(&mut self, gold: &WordLanguage<Gold>) {
        self.gap.deletions.push(gold.language.bucket.clone());
    }

    pub(in crate::compare) fn inserted(&mut self, attribution: &InsertionAttribution<'_>) {
        self.gap.insertions.push(match attribution {
            InsertionAttribution::InGold(site) => {
                PendingInsertion::Attributed(site.owner().language.language.bucket.clone())
            }
            InsertionAttribution::NoGoldWords => PendingInsertion::Unattributed,
        });
    }

    pub(in crate::compare) fn placed_utterance(
        &mut self,
        gold: &SidedLanguage<Gold>,
        main: &SidedLanguage<Main>,
    ) {
        self.scores.utterances.placed.record(gold, main);
    }

    pub(in crate::compare) fn unplaced_utterance(&mut self, gold: &SidedLanguage<Gold>) {
        *self
            .scores
            .utterances
            .unplaced
            .entry(gold.bucket.clone())
            .or_default() += 1;
    }

    /// Pair the open gap's deletions with its insertions, in order.
    ///
    /// Called at every match and at the end, so errors never pair across a
    /// match.
    fn close_gap(&mut self) {
        let OpenGap {
            deletions,
            insertions,
        } = std::mem::take(&mut self.gap);
        let paired = deletions.len().min(insertions.len());

        let mut deletions = deletions.into_iter();
        for language in deletions.by_ref().take(paired) {
            self.counts(&language).substitutions += 1;
        }
        for language in deletions {
            self.counts(&language).deletions += 1;
        }
        for attribution in insertions.into_iter().skip(paired) {
            match attribution {
                PendingInsertion::Attributed(language) => self.counts(&language).insertions += 1,
                PendingInsertion::Unattributed => self.scores.unattributed_insertions += 1,
            }
        }
    }

    pub(in crate::compare) fn finish(mut self) -> LanguageScores {
        self.close_gap();
        self.scores
    }

    fn counts(&mut self, language: &LanguageBucket) -> &mut LanguageErrorCounts {
        self.scores.by_language.entry(language.clone()).or_default()
    }
}
