//! Comparison of two already-produced transcripts.
//!
//! Unlike [`super::engine::compare`], this surface has no gold side.  Speaker
//! correspondence is established first, and metrics are only reported for an
//! established mapping.  It serves any comparison where neither producer is
//! authoritative, including one whose other side is a human transcript.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::PathBuf;

use serde::Serialize;
use talkbank_model::ErrorCollector;
use talkbank_model::alignment::helpers::PositionalDomain;
use talkbank_model::model::ChatFile;
use talkbank_parser::TreeSitterParser;

use crate::dp_align::{self, AlignResult, MatchMode};
use crate::extract;

use super::artifact::{ComparisonSubject, ValidatedTranscriptionPlan};
use super::engine::{conform_with_mapping, is_punct_or_filler};

/// Version of the cross-run report artifact schema.
pub const CROSS_RUN_REPORT_SCHEMA_VERSION: u32 = 1;

/// Errors while materializing a validated cross-run result.
#[derive(Debug, thiserror::Error)]
pub enum CrossRunReportError {
    /// JSON serialization failed.
    #[error("cross-run report JSON serialization failed: {0}")]
    Json(#[from] serde_json::Error),
    /// CSV serialization failed.
    #[error("cross-run report CSV serialization failed: {0}")]
    Csv(#[from] csv::Error),
    /// The CSV writer could not be finalized.
    #[error("cross-run report CSV writer failed: {0}")]
    Io(#[from] std::io::Error),
    /// The in-memory CSV was not UTF-8.
    #[error("cross-run report CSV was not UTF-8: {0}")]
    Utf8(#[from] std::string::FromUtf8Error),
    /// An artifact could not be read as UTF-8.
    #[error("cannot read comparison artifact {path}: {detail}")]
    ArtifactRead {
        /// Artifact path that failed.
        path: PathBuf,
        /// Underlying I/O or UTF-8 error text.
        detail: String,
    },
    /// CHAT parsing produced diagnostics, so no comparison was emitted.
    #[error("comparison artifact {path} did not parse cleanly: {diagnostics}")]
    ArtifactParse {
        /// Artifact path that failed.
        path: PathBuf,
        /// Parser diagnostics.
        diagnostics: String,
    },
    /// The parser could not be initialized.
    #[error("cannot initialize CHAT parser: {0}")]
    ParserInit(String),
}

/// Versioned report envelope with the run identities that produced it.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CrossRunTranscriptionReportArtifact {
    /// Report schema version.
    schema_version: u32,
    /// Comparison subject.
    subject: ComparisonSubject,
    /// Left run identity.
    left_run_id: String,
    /// Right run identity.
    right_run_id: String,
    /// Typed comparison result.
    report: CrossRunTranscriptionReport,
}

/// One explicitly paired artifact's transcription result.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ArtifactTranscriptionComparison {
    /// Left-run relative artifact path.
    left_artifact: String,
    /// Right-run relative artifact path.
    right_artifact: String,
    /// Typed speaker-aware result.
    report: CrossRunTranscriptionReport,
}

impl ArtifactTranscriptionComparison {
    /// Left-run relative artifact path.
    pub fn left_artifact(&self) -> &str {
        &self.left_artifact
    }

    /// Right-run relative artifact path.
    pub fn right_artifact(&self) -> &str {
        &self.right_artifact
    }

    /// Typed speaker-aware result.
    pub fn report(&self) -> &CrossRunTranscriptionReport {
        &self.report
    }
}

/// Result of executing every pair in a validated transcription plan.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CrossRunTranscriptionPlanReport {
    /// Report schema version.
    schema_version: u32,
    /// Comparison subject.
    subject: ComparisonSubject,
    /// Left run identity.
    left_run_id: String,
    /// Right run identity.
    right_run_id: String,
    /// Results in declared pair order.
    comparisons: Vec<ArtifactTranscriptionComparison>,
}

impl CrossRunTranscriptionPlanReport {
    /// Results in declared pair order.
    pub fn comparisons(&self) -> &[ArtifactTranscriptionComparison] {
        &self.comparisons
    }
}

/// A one-to-one mapping from speakers in the left run to speakers in the right.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SpeakerMap {
    /// Left-run speaker to right-run speaker assignments.
    assignments: BTreeMap<String, String>,
}

/// Errors constructing a one-to-one explicit speaker map.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SpeakerMapBuildError {
    /// A map must identify at least one speaker pair.
    #[error("speaker map must contain at least one assignment")]
    Empty,
    /// A left speaker label is empty.
    #[error("speaker map contains an empty left speaker label")]
    EmptyLeftSpeaker,
    /// A right speaker label is empty.
    #[error("speaker map contains an empty right speaker label")]
    EmptyRightSpeaker,
    /// Two left speakers target the same right speaker.
    #[error("speaker map assigns right speaker {speaker:?} more than once")]
    DuplicateRightSpeaker {
        /// Reused right speaker label.
        speaker: String,
    },
}

/// Errors applying a structurally valid map to a particular transcript pair.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SpeakerMapApplicationError {
    /// A mapped left speaker is absent from the left transcript.
    #[error("mapped left speaker {speaker:?} is absent from the left transcript")]
    UnknownLeftSpeaker {
        /// Missing left speaker label.
        speaker: String,
    },
    /// A mapped right speaker is absent from the right transcript.
    #[error("mapped right speaker {speaker:?} is absent from the right transcript")]
    UnknownRightSpeaker {
        /// Missing right speaker label.
        speaker: String,
    },
}

impl SpeakerMap {
    fn new(assignments: BTreeMap<String, String>) -> Option<Self> {
        Self::try_from_assignments(assignments).ok()
    }

    /// Construct a non-empty, injective speaker map.
    pub fn try_from_assignments(
        assignments: BTreeMap<String, String>,
    ) -> Result<Self, SpeakerMapBuildError> {
        if assignments.is_empty() {
            return Err(SpeakerMapBuildError::Empty);
        }
        let mut right_speakers = BTreeSet::new();
        for (left, right) in &assignments {
            if left.is_empty() {
                return Err(SpeakerMapBuildError::EmptyLeftSpeaker);
            }
            if right.is_empty() {
                return Err(SpeakerMapBuildError::EmptyRightSpeaker);
            }
            if !right_speakers.insert(right.as_str()) {
                return Err(SpeakerMapBuildError::DuplicateRightSpeaker {
                    speaker: right.clone(),
                });
            }
        }
        Ok(Self { assignments })
    }

    /// One-to-one left-to-right speaker assignments.
    pub fn assignments(&self) -> &BTreeMap<String, String> {
        &self.assignments
    }
}

/// A possible speaker mapping and its evidence score.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SpeakerMapCandidate {
    /// Proposed one-to-one assignment.
    map: SpeakerMap,
    /// Number of order-insensitive word overlaps supporting the assignment.
    overlap: usize,
}

impl SpeakerMapCandidate {
    /// Proposed one-to-one assignment.
    pub fn map(&self) -> &SpeakerMap {
        &self.map
    }

    /// Number of order-insensitive word overlaps supporting the assignment.
    pub fn overlap(&self) -> usize {
        self.overlap
    }
}

/// A non-empty set of equally supported speaker mappings.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AmbiguousSpeakerMaps(Vec<SpeakerMapCandidate>);

impl AmbiguousSpeakerMaps {
    /// Candidate mappings. This slice is guaranteed non-empty.
    pub fn candidates(&self) -> &[SpeakerMapCandidate] {
        &self.0
    }
}

/// Outcome of attempting to establish speaker correspondence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum SpeakerCorrespondence {
    /// Exactly one best mapping was found.
    Established(SpeakerMap),
    /// There was no usable speaker evidence.
    Unavailable,
    /// Multiple equally good mappings remain possible.
    Ambiguous(AmbiguousSpeakerMaps),
}

/// A quantity measured on both sides of a comparison.
///
/// Three pairs of `left_*`/`right_*` fields used to sit flat in
/// `AgreementMetrics`, which is what pushed it over the wide-struct threshold
/// and, more to the point, left the symmetry implicit: nothing tied
/// `left_words` to `right_words` except their names, and a two-field struct
/// cannot be half-updated the way two loose fields can.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub(in crate::compare) struct SideCounts {
    /// The left run.
    pub(in crate::compare) left: usize,
    /// The right run.
    pub(in crate::compare) right: usize,
}

/// Agreement metrics for one established speaker pair.
///
/// The serialized shape is a CONTRACT with `compare-runs --format csv`, which
/// reads it by JSON pointer, and a pointer that misses yields an empty cell at
/// runtime rather than a compile error. `serialized_keys_match_the_csv_reader`
/// below pins it. A mirror "wire struct" was tried first and was worse: it
/// duplicated all eleven values, needed a test to hold the two copies in step,
/// and was itself wide enough to fail the wide-struct audit.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AgreementMetrics {
    /// Order-insensitive residual-error numerator.
    ///
    /// The one input that is not derivable from the alignment itself.
    cwer_numerator: usize,
    /// Word totals for each run.
    words: SideCounts,
    /// Matches, insertions and deletions from ONE alignment.
    ///
    /// Held as the tally rather than three loose counts so that a mix of
    /// numbers from different alignments is unrepresentable.
    tally: AlignmentTally,
    /// De-identification/exclusion tokens omitted from scoring, per side.
    excluded_tokens: SideCounts,
}

/// Ordered-alignment tallies for one speaker pair.
///
/// A struct rather than three `usize` arguments: matches, insertions and
/// deletions are counts of different things, and positionally they are
/// indistinguishable, so a transposition would be silent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
struct AlignmentTally {
    matches: usize,
    insertions: usize,
    deletions: usize,
}

impl AgreementMetrics {
    /// Build metrics from the two word sequences and their tallies.
    ///
    /// Every dependent value is derived HERE and nowhere else. `wer_numerator`
    /// is `insertions + deletions`, the word totals come from the sequences
    /// themselves, and each rate is its numerator over `left_words`. That
    /// matters because the previous struct literal stated the numerator twice
    /// and the denominator twice, in independent expressions, so a later edit
    /// to one could leave the other stale and nothing would notice. Callers
    /// cannot build this struct literally: its fields are private to the
    /// module and this is the only constructor.
    fn new(
        left: &[String],
        right: &[String],
        tally: AlignmentTally,
        cwer_numerator: usize,
        excluded: SideCounts,
    ) -> Self {
        let left_words = left.len();
        Self {
            cwer_numerator,
            words: SideCounts {
                left: left_words,
                right: right.len(),
            },
            tally,
            excluded_tokens: excluded,
        }
    }

    /// A numerator over the left-side word count.
    ///
    /// Absent rather than zero when there is nothing to divide by: a rate of
    /// 0.0 would read as perfect agreement on an empty side. Both public
    /// rates go through here, so that rule has one statement.
    fn rate(&self, numerator: usize) -> Option<f64> {
        (self.words.left > 0).then(|| numerator as f64 / self.words.left as f64)
    }

    /// Ordered agreement-error numerator.
    ///
    /// DERIVED, not stored: it is exactly `insertions + deletions`, and a
    /// stored copy is a second representation of a fact the struct already
    /// holds, free to drift from it.
    pub fn wer_numerator(&self) -> usize {
        self.tally.insertions + self.tally.deletions
    }

    /// Order-insensitive agreement-error numerator.
    pub fn cwer_numerator(&self) -> usize {
        self.cwer_numerator
    }

    /// Number of words in the left run.
    pub fn left_words(&self) -> usize {
        self.words.left
    }

    /// Number of words in the right run.
    pub fn right_words(&self) -> usize {
        self.words.right
    }

    /// Number of ordered matches.
    pub fn matches(&self) -> usize {
        self.tally.matches
    }

    /// Words only in the left run.
    pub fn insertions(&self) -> usize {
        self.tally.insertions
    }

    /// Words only in the right run.
    pub fn deletions(&self) -> usize {
        self.tally.deletions
    }

    /// Ordered agreement error rate, absent for a zero-word left side.
    pub fn wer_rate(&self) -> Option<f64> {
        self.rate(self.wer_numerator())
    }
    /// Order-insensitive agreement error rate, absent for a zero-word left side.
    pub fn cwer_rate(&self) -> Option<f64> {
        self.rate(self.cwer_numerator)
    }
    /// Excluded left token count.
    pub fn excluded_left_tokens(&self) -> usize {
        self.excluded_tokens.left
    }
    /// Excluded right token count.
    pub fn excluded_right_tokens(&self) -> usize {
        self.excluded_tokens.right
    }
}

/// Metrics and labels for one established speaker pair.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SpeakerAgreement {
    /// Left-run speaker label.
    left_speaker: String,
    /// Right-run speaker label.
    right_speaker: String,
    /// Pairwise agreement metrics.
    metrics: AgreementMetrics,
}

impl SpeakerAgreement {
    /// Left-run speaker label.
    pub fn left_speaker(&self) -> &str {
        &self.left_speaker
    }

    /// Right-run speaker label.
    pub fn right_speaker(&self) -> &str {
        &self.right_speaker
    }

    /// Pairwise agreement metrics.
    pub fn metrics(&self) -> &AgreementMetrics {
        &self.metrics
    }
}

/// Typed cross-run transcription report.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CrossRunTranscriptionReport {
    /// Result of speaker correspondence inference.
    correspondence: SpeakerCorrespondence,
    /// Metrics for mapped speakers; empty unless correspondence is established.
    speaker_agreements: Vec<SpeakerAgreement>,
    /// Left speakers that have no mapped right speaker.
    unmatched_left_speakers: Vec<String>,
    /// Right speakers that have no mapped left speaker.
    unmatched_right_speakers: Vec<String>,
}

impl CrossRunTranscriptionReport {
    /// Result of speaker correspondence inference.
    pub fn correspondence(&self) -> &SpeakerCorrespondence {
        &self.correspondence
    }

    /// Metrics for mapped speakers; empty unless correspondence is established.
    pub fn speaker_agreements(&self) -> &[SpeakerAgreement] {
        &self.speaker_agreements
    }

    /// Left speakers with no mapped right speaker.
    pub fn unmatched_left_speakers(&self) -> &[String] {
        &self.unmatched_left_speakers
    }

    /// Right speakers with no mapped left speaker.
    pub fn unmatched_right_speakers(&self) -> &[String] {
        &self.unmatched_right_speakers
    }
}

/// Compare two transcript artifacts without treating either as gold.
pub fn compare_transcripts_by_speaker(
    left: &ChatFile,
    right: &ChatFile,
) -> CrossRunTranscriptionReport {
    let left_words = speaker_words(left, &BTreeSet::new());
    let right_words = speaker_words(right, &BTreeSet::new());
    let correspondence = establish_speaker_correspondence(&left_words, &right_words);
    report_from_correspondence(
        left_words,
        right_words,
        correspondence,
        &BTreeMap::new(),
        &BTreeMap::new(),
    )
}

/// Compare transcripts while separately accounting for configured placeholder tokens.
pub fn compare_transcripts_with_exclusions(
    left: &ChatFile,
    right: &ChatFile,
    exclusions: &BTreeSet<String>,
    explicit_map: Option<SpeakerMap>,
) -> Result<CrossRunTranscriptionReport, SpeakerMapApplicationError> {
    let (left_words, left_excluded) = speaker_words_and_excluded(left, exclusions);
    let (right_words, right_excluded) = speaker_words_and_excluded(right, exclusions);
    let correspondence = explicit_map.map_or_else(
        || establish_speaker_correspondence(&left_words, &right_words),
        SpeakerCorrespondence::Established,
    );
    validate_correspondence(&left_words, &right_words, &correspondence)?;
    Ok(report_from_correspondence(
        left_words,
        right_words,
        correspondence,
        &left_excluded,
        &right_excluded,
    ))
}

/// Compare using a human-adjudicated, structurally valid speaker map.
///
/// Partial maps are intentional: they allow a caller to score only an adult
/// or other reviewed subset while unmatched speakers remain explicit in the
/// report. Every mapped label must exist on its declared side.
pub fn compare_transcripts_with_speaker_map(
    left: &ChatFile,
    right: &ChatFile,
    map: SpeakerMap,
) -> Result<CrossRunTranscriptionReport, SpeakerMapApplicationError> {
    let left_words = speaker_words(left, &BTreeSet::new());
    let right_words = speaker_words(right, &BTreeSet::new());
    for left_speaker in map.assignments.keys() {
        if !left_words.contains_key(left_speaker) {
            return Err(SpeakerMapApplicationError::UnknownLeftSpeaker {
                speaker: left_speaker.clone(),
            });
        }
    }
    for right_speaker in map.assignments.values() {
        if !right_words.contains_key(right_speaker) {
            return Err(SpeakerMapApplicationError::UnknownRightSpeaker {
                speaker: right_speaker.clone(),
            });
        }
    }
    Ok(report_from_correspondence(
        left_words,
        right_words,
        SpeakerCorrespondence::Established(map),
        &BTreeMap::new(),
        &BTreeMap::new(),
    ))
}

fn validate_correspondence(
    left: &BTreeMap<String, Vec<String>>,
    right: &BTreeMap<String, Vec<String>>,
    correspondence: &SpeakerCorrespondence,
) -> Result<(), SpeakerMapApplicationError> {
    let SpeakerCorrespondence::Established(map) = correspondence else {
        return Ok(());
    };
    for speaker in map.assignments.keys() {
        if !left.contains_key(speaker) {
            return Err(SpeakerMapApplicationError::UnknownLeftSpeaker {
                speaker: speaker.clone(),
            });
        }
    }
    for speaker in map.assignments.values() {
        if !right.contains_key(speaker) {
            return Err(SpeakerMapApplicationError::UnknownRightSpeaker {
                speaker: speaker.clone(),
            });
        }
    }
    Ok(())
}

fn report_from_correspondence(
    left_words: BTreeMap<String, Vec<String>>,
    right_words: BTreeMap<String, Vec<String>>,
    correspondence: SpeakerCorrespondence,
    left_excluded: &BTreeMap<String, usize>,
    right_excluded: &BTreeMap<String, usize>,
) -> CrossRunTranscriptionReport {
    let SpeakerCorrespondence::Established(map) = &correspondence else {
        return CrossRunTranscriptionReport {
            correspondence,
            speaker_agreements: Vec::new(),
            unmatched_left_speakers: left_words.keys().cloned().collect(),
            unmatched_right_speakers: right_words.keys().cloned().collect(),
        };
    };

    let mapped_right: BTreeSet<&str> = map.assignments.values().map(String::as_str).collect();
    let speaker_agreements = map
        .assignments
        .iter()
        .filter_map(|(left_speaker, right_speaker)| {
            let left_tokens = left_words.get(left_speaker)?;
            let right_tokens = right_words.get(right_speaker)?;
            Some(SpeakerAgreement {
                left_speaker: left_speaker.clone(),
                right_speaker: right_speaker.clone(),
                metrics: agreement_metrics(
                    left_tokens,
                    right_tokens,
                    left_excluded.get(left_speaker).copied().unwrap_or(0),
                    right_excluded.get(right_speaker).copied().unwrap_or(0),
                ),
            })
        })
        .collect();
    let unmatched_left_speakers = left_words
        .keys()
        .filter(|speaker| !map.assignments.contains_key(*speaker))
        .cloned()
        .collect();
    let unmatched_right_speakers = right_words
        .keys()
        .filter(|speaker| !mapped_right.contains(speaker.as_str()))
        .cloned()
        .collect();

    CrossRunTranscriptionReport {
        correspondence,
        speaker_agreements,
        unmatched_left_speakers,
        unmatched_right_speakers,
    }
}

/// Execute a validated transcription plan over its declared artifact pairs.
///
/// The plan's typestate proves that each path exists and its bytes match the
/// manifest before this function reads anything. Parsing is another explicit
/// gate: a pair with diagnostics fails the whole operation instead of
/// producing a partial or silently repaired comparison.
pub fn compare_validated_transcription_plan(
    plan: &ValidatedTranscriptionPlan,
) -> Result<CrossRunTranscriptionPlanReport, CrossRunReportError> {
    let parser = TreeSitterParser::new()
        .map_err(|error| CrossRunReportError::ParserInit(error.to_string()))?;
    let mut comparisons = Vec::with_capacity(plan.artifact_pairs().len());
    for pair in plan.artifact_pairs() {
        let left_path = plan.runs()[0].artifacts().path().join(pair.left().as_str());
        let right_path = plan.runs()[1]
            .artifacts()
            .path()
            .join(pair.right().as_str());
        let left = parse_artifact(&parser, &left_path)?;
        let right = parse_artifact(&parser, &right_path)?;
        comparisons.push(ArtifactTranscriptionComparison {
            left_artifact: pair.left().as_str().to_string(),
            right_artifact: pair.right().as_str().to_string(),
            report: compare_transcripts_by_speaker(&left, &right),
        });
    }
    Ok(CrossRunTranscriptionPlanReport {
        schema_version: CROSS_RUN_REPORT_SCHEMA_VERSION,
        subject: ComparisonSubject::Transcription,
        left_run_id: plan.runs()[0].manifest().run_id().to_string(),
        right_run_id: plan.runs()[1].manifest().run_id().to_string(),
        comparisons,
    })
}

fn parse_artifact(
    parser: &TreeSitterParser,
    path: &std::path::Path,
) -> Result<ChatFile, CrossRunReportError> {
    let bytes = std::fs::read(path).map_err(|error| CrossRunReportError::ArtifactRead {
        path: path.to_path_buf(),
        detail: error.to_string(),
    })?;
    let text = String::from_utf8(bytes).map_err(|error| CrossRunReportError::ArtifactRead {
        path: path.to_path_buf(),
        detail: error.to_string(),
    })?;
    let errors = ErrorCollector::new();
    let chat_file = parser.parse_chat_file_streaming(&text, &errors);
    let diagnostics = errors.into_vec();
    if diagnostics.is_empty() {
        Ok(chat_file)
    } else {
        Err(CrossRunReportError::ArtifactParse {
            path: path.to_path_buf(),
            diagnostics: format!("{diagnostics:?}"),
        })
    }
}

/// Serialize a report together with its validated run provenance.
pub fn serialize_cross_run_json(
    plan: &ValidatedTranscriptionPlan,
    report: &CrossRunTranscriptionReport,
) -> Result<String, CrossRunReportError> {
    let artifact = CrossRunTranscriptionReportArtifact {
        schema_version: CROSS_RUN_REPORT_SCHEMA_VERSION,
        subject: ComparisonSubject::Transcription,
        left_run_id: plan.runs()[0].manifest().run_id().to_string(),
        right_run_id: plan.runs()[1].manifest().run_id().to_string(),
        report: report.clone(),
    };
    Ok(serde_json::to_string_pretty(&artifact)?)
}

/// Serialize one stable CSV row per established speaker pair.
pub fn serialize_cross_run_csv(
    plan: &ValidatedTranscriptionPlan,
    report: &CrossRunTranscriptionReport,
) -> Result<String, CrossRunReportError> {
    let mut writer = csv::Writer::from_writer(Vec::new());
    writer.write_record([
        "schema_version",
        "left_run_id",
        "right_run_id",
        "left_speaker",
        "right_speaker",
        "wer_numerator",
        "cwer_numerator",
        "left_words",
        "right_words",
        "matches",
        "insertions",
        "deletions",
    ])?;
    for agreement in &report.speaker_agreements {
        let metrics = &agreement.metrics;
        writer.write_record([
            CROSS_RUN_REPORT_SCHEMA_VERSION.to_string(),
            plan.runs()[0].manifest().run_id().to_string(),
            plan.runs()[1].manifest().run_id().to_string(),
            agreement.left_speaker.clone(),
            agreement.right_speaker.clone(),
            metrics.wer_numerator().to_string(),
            metrics.cwer_numerator().to_string(),
            metrics.left_words().to_string(),
            metrics.right_words().to_string(),
            metrics.matches().to_string(),
            metrics.insertions().to_string(),
            metrics.deletions().to_string(),
        ])?;
    }
    Ok(String::from_utf8(
        writer.into_inner().map_err(|error| error.into_error())?,
    )?)
}

fn speaker_words(file: &ChatFile, exclusions: &BTreeSet<String>) -> BTreeMap<String, Vec<String>> {
    speaker_words_and_excluded(file, exclusions).0
}

fn speaker_words_and_excluded(
    file: &ChatFile,
    exclusions: &BTreeSet<String>,
) -> (BTreeMap<String, Vec<String>>, BTreeMap<String, usize>) {
    let extracted = extract::extract_words(file, PositionalDomain::Mor);
    let mut result = BTreeMap::new();
    let mut excluded = BTreeMap::new();
    for utterance in extracted {
        let words = result
            .entry(utterance.speaker.as_str().to_string())
            .or_insert_with(Vec::new);
        for word in utterance.words {
            let text = word.text.as_str();
            if exclusions.contains(text) || exclusions.contains(&text.to_lowercase()) {
                *excluded
                    .entry(utterance.speaker.as_str().to_string())
                    .or_insert(0) += 1;
                continue;
            }
            if is_punct_or_filler(text) {
                continue;
            }
            let (conformed, _) = conform_with_mapping(&[text.to_string()]);
            words.extend(conformed);
        }
    }
    (result, excluded)
}

fn establish_speaker_correspondence(
    left: &BTreeMap<String, Vec<String>>,
    right: &BTreeMap<String, Vec<String>>,
) -> SpeakerCorrespondence {
    if left.is_empty() || right.is_empty() {
        return SpeakerCorrespondence::Unavailable;
    }

    let left_names: Vec<String> = left.keys().cloned().collect();
    let right_names: Vec<String> = right.keys().cloned().collect();
    let overlap: Vec<Vec<usize>> = left_names
        .iter()
        .map(|left_name| {
            right_names
                .iter()
                .map(|right_name| multiset_overlap(&left[left_name], &right[right_name]))
                .collect()
        })
        .collect();

    let mut search = SpeakerMapSearch {
        left_names: &left_names,
        right_names: &right_names,
        overlap: &overlap,
        candidates: Vec::new(),
    };
    search.enumerate(0, &mut BTreeSet::new(), &mut BTreeMap::new(), 0);
    let mut candidates = search.candidates;
    let best = candidates.iter().map(|candidate| candidate.overlap).max();
    let Some(best_overlap) = best else {
        return SpeakerCorrespondence::Unavailable;
    };
    if best_overlap == 0 {
        return SpeakerCorrespondence::Unavailable;
    }
    candidates.retain(|candidate| candidate.overlap == best_overlap);
    candidates.sort_by(|left, right| left.map.assignments.cmp(&right.map.assignments));
    candidates.dedup_by(|left, right| left.map == right.map);
    if candidates.len() == 1 {
        SpeakerCorrespondence::Established(candidates.remove(0).map)
    } else {
        debug_assert!(!candidates.is_empty());
        SpeakerCorrespondence::Ambiguous(AmbiguousSpeakerMaps(candidates))
    }
}

struct SpeakerMapSearch<'a> {
    left_names: &'a [String],
    right_names: &'a [String],
    overlap: &'a [Vec<usize>],
    candidates: Vec<SpeakerMapCandidate>,
}

impl SpeakerMapSearch<'_> {
    fn enumerate(
        &mut self,
        index: usize,
        used_right: &mut BTreeSet<usize>,
        assignments: &mut BTreeMap<String, String>,
        score: usize,
    ) {
        if index == self.left_names.len() {
            if let Some(map) = SpeakerMap::new(assignments.clone()) {
                self.candidates.push(SpeakerMapCandidate {
                    map,
                    overlap: score,
                });
            }
            return;
        }
        for right_index in 0..self.right_names.len() {
            if used_right.contains(&right_index) {
                continue;
            }
            used_right.insert(right_index);
            assignments.insert(
                self.left_names[index].clone(),
                self.right_names[right_index].clone(),
            );
            self.enumerate(
                index + 1,
                used_right,
                assignments,
                score + self.overlap[index][right_index],
            );
            assignments.remove(&self.left_names[index]);
            used_right.remove(&right_index);
        }
        // A larger left speaker set may have speakers with no counterpart.
        // This branch is explicit so the report can expose unmatched speakers
        // rather than forcing a false assignment.
        if self.left_names.len() > self.right_names.len() || self.right_names.is_empty() {
            self.enumerate(index + 1, used_right, assignments, score);
        }
    }
}

fn multiset_overlap(left: &[String], right: &[String]) -> usize {
    let mut counts = HashMap::new();
    for token in right {
        *counts.entry(token.to_lowercase()).or_insert(0usize) += 1;
    }
    left.iter().fold(0, |total, token| {
        let key = token.to_lowercase();
        let Some(count) = counts.get_mut(&key) else {
            return total;
        };
        if *count == 0 {
            total
        } else {
            *count -= 1;
            total + 1
        }
    })
}

fn agreement_metrics(
    left: &[String],
    right: &[String],
    excluded_left_tokens: usize,
    excluded_right_tokens: usize,
) -> AgreementMetrics {
    let alignment = dp_align::align(left, right, MatchMode::CaseInsensitive);
    let mut matches = 0;
    let mut insertions = 0;
    let mut deletions = 0;
    for item in alignment {
        match item {
            AlignResult::Match { .. } => matches += 1,
            AlignResult::ExtraPayload { .. } => insertions += 1,
            AlignResult::ExtraReference { .. } => deletions += 1,
        }
    }
    let overlap = multiset_overlap(left, right);
    let cwer_numerator = (left.len() - overlap) + (right.len() - overlap);
    AgreementMetrics::new(
        left,
        right,
        AlignmentTally {
            matches,
            insertions,
            deletions,
        },
        cwer_numerator,
        SideCounts {
            left: excluded_left_tokens,
            right: excluded_right_tokens,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compare::{
        ArtifactPair, ComparisonPlan, MachineIdentity, PairingPolicy, ProducedRun,
        RelativeArtifactPath, RunArtifactRoot, RunIdentity, RunManifest, ValidatedComparisonPlan,
    };
    use std::collections::BTreeMap;
    use talkbank_model::ErrorCollector;
    use talkbank_parser::TreeSitterParser;

    fn parse(text: &str) -> ChatFile {
        let parser = TreeSitterParser::new().unwrap();
        let errors = ErrorCollector::new();
        parser.parse_chat_file_streaming(text, &errors)
    }

    fn chat(left: &str, right: &str) -> String {
        format!(
            "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tPAR0 A, PAR1 B\n@ID:\teng|test|PAR0|3;|female|||A|||\n@ID:\teng|test|PAR1||female|||B|||\n*PAR0:\t{left} .\n*PAR1:\t{right} .\n@End\n"
        )
    }

    #[test]
    fn establishes_unique_mapping_and_reports_ordered_and_unordered_counts() {
        let left = parse(&chat("alpha beta", "gamma delta"));
        let right = parse(&chat("gamma delta", "alpha beta"));
        let report = compare_transcripts_by_speaker(&left, &right);
        let SpeakerCorrespondence::Established(map) = report.correspondence() else {
            panic!("expected a unique speaker mapping");
        };
        assert_eq!(map.assignments()["PAR0"], "PAR1");
        assert_eq!(map.assignments()["PAR1"], "PAR0");
        assert!(report.speaker_agreements().iter().all(|agreement| {
            agreement.metrics().wer_numerator() == 0 && agreement.metrics().cwer_numerator() == 0
        }));
    }

    #[test]
    fn refuses_to_invent_mapping_when_evidence_is_tied() {
        let left = parse(&chat("hello", "hello"));
        let right = parse(&chat("hello", "hello"));
        let report = compare_transcripts_by_speaker(&left, &right);
        assert!(matches!(
            report.correspondence(),
            SpeakerCorrespondence::Ambiguous(_)
        ));
        assert!(report.speaker_agreements().is_empty());
    }

    #[test]
    fn explicit_speaker_map_rejects_duplicate_right_speaker() {
        let assignments = BTreeMap::from([
            ("PAR0".to_string(), "MOT".to_string()),
            ("PAR1".to_string(), "MOT".to_string()),
        ]);
        assert!(matches!(
            SpeakerMap::try_from_assignments(assignments),
            Err(SpeakerMapBuildError::DuplicateRightSpeaker { .. })
        ));
    }

    #[test]
    fn explicit_speaker_map_is_checked_against_both_transcripts() {
        let left = parse(&chat("alpha", "beta"));
        let right = parse(&chat("alpha", "beta"));
        let map = SpeakerMap::try_from_assignments(BTreeMap::from([(
            "PAR0".to_string(),
            "MISSING".to_string(),
        )]))
        .unwrap();
        assert!(matches!(
            compare_transcripts_with_speaker_map(&left, &right, map),
            Err(SpeakerMapApplicationError::UnknownRightSpeaker { .. })
        ));
    }

    #[test]
    fn explicit_partial_map_scores_only_the_adjudicated_speaker_pair() {
        let left = parse(&chat("same", "same"));
        let right = parse(&chat("same", "same"));
        let map = SpeakerMap::try_from_assignments(BTreeMap::from([(
            "PAR0".to_string(),
            "PAR1".to_string(),
        )]))
        .unwrap();
        let report = compare_transcripts_with_speaker_map(&left, &right, map).unwrap();
        assert_eq!(report.speaker_agreements().len(), 1);
        assert_eq!(report.unmatched_left_speakers(), &["PAR1".to_string()]);
        assert_eq!(report.unmatched_right_speakers(), &["PAR0".to_string()]);
    }

    #[test]
    fn serializers_require_verified_provenance_and_emit_stable_fields() {
        let left_dir = tempfile::tempdir().unwrap();
        let right_dir = tempfile::tempdir().unwrap();
        std::fs::write(
            left_dir.path().join("result.cha"),
            chat("alpha", "beta").as_bytes(),
        )
        .unwrap();
        std::fs::write(
            right_dir.path().join("result.cha"),
            chat("alpha", "beta").as_bytes(),
        )
        .unwrap();
        let identity = |implementation: &str| {
            RunIdentity::Machine(
                MachineIdentity::new(
                    implementation.to_string(),
                    "transcribe".to_string(),
                    "test".to_string(),
                )
                .unwrap(),
            )
        };
        let left = RunManifest::from_artifact_root(
            left_dir.path(),
            "left".to_string(),
            identity("ours"),
            "source".to_string(),
            BTreeMap::new(),
        )
        .unwrap();
        let right = RunManifest::from_artifact_root(
            right_dir.path(),
            "right".to_string(),
            identity("other-impl"),
            "source".to_string(),
            BTreeMap::new(),
        )
        .unwrap();
        let plan = ComparisonPlan {
            subject: ComparisonSubject::Transcription,
            runs: [
                ProducedRun {
                    manifest: left,
                    artifacts: RunArtifactRoot::new(left_dir.path().to_path_buf()),
                },
                ProducedRun {
                    manifest: right,
                    artifacts: RunArtifactRoot::new(right_dir.path().to_path_buf()),
                },
            ],
            pairing: PairingPolicy::SameSourceChat,
            artifact_pairs: vec![ArtifactPair::new(
                RelativeArtifactPath::new("result.cha").unwrap(),
                RelativeArtifactPath::new("result.cha").unwrap(),
            )],
            output: tempfile::tempdir().unwrap().path().to_path_buf(),
        };
        let ValidatedComparisonPlan::Transcription(validated) = plan.validate().unwrap() else {
            panic!("transcription input must produce transcription typestate");
        };
        let plan_report = compare_validated_transcription_plan(&validated).unwrap();
        assert_eq!(plan_report.comparisons().len(), 1);
        assert_eq!(plan_report.comparisons()[0].left_artifact(), "result.cha");
        let report = compare_transcripts_by_speaker(
            &parse(&chat("alpha", "beta")),
            &parse(&chat("alpha", "beta")),
        );
        let json = serialize_cross_run_json(&validated, &report).unwrap();
        assert!(json.contains("\"schema_version\": 1"));
        assert!(json.contains("\"left_run_id\": \"left\""));
        let csv = serialize_cross_run_csv(&validated, &report).unwrap();
        assert!(csv.starts_with("schema_version,left_run_id,right_run_id,left_speaker"));
    }
}

#[cfg(test)]
mod agreement_metrics_wire_tests {
    use super::{AgreementMetrics, AlignmentTally, SideCounts};

    /// The serialized keys are a CONTRACT with `compare-runs --format csv`,
    /// which reads them by JSON pointer.
    ///
    /// A pointer that misses produces an EMPTY CELL at runtime, not a compile
    /// error, so nothing in the type system connects this struct to that
    /// reader. Regrouping the internal fields once silently emptied eleven of
    /// sixteen columns. This test is the connection: change the wire shape and
    /// it fails here, next to the reason.
    #[test]
    fn serialized_keys_match_the_csv_reader() {
        let metrics = AgreementMetrics::new(
            &["a".to_string(), "b".to_string()],
            &["a".to_string()],
            AlignmentTally {
                matches: 1,
                insertions: 1,
                deletions: 0,
            },
            1,
            SideCounts { left: 3, right: 4 },
        );

        let json = serde_json::to_value(&metrics).expect("metrics serialize");
        let object = json.as_object().expect("a flat object, not a nested one");

        // Exactly the pointers `cli/compare_runs_cmd.rs` reads. The nested
        // ones are spelled as pointers because that is how the reader spells
        // them; if this list and that reader ever disagree, the CSV silently
        // emits empty cells, which is the failure this pins.
        for pointer in [
            "/cwer_numerator",
            "/words/left",
            "/words/right",
            "/tally/matches",
            "/tally/insertions",
            "/tally/deletions",
            "/excluded_tokens/left",
            "/excluded_tokens/right",
        ] {
            assert!(
                json.pointer(pointer).is_some(),
                "CSV reader needs {pointer:?}: {json}"
            );
        }
        assert_eq!(
            object.len(),
            4,
            "an unexpected top-level key means the wire shape moved: {json}"
        );

        // The derived values the reader computes rather than reads.
        assert_eq!(metrics.wer_numerator(), 1);
        assert_eq!(
            json.pointer("/words/left").and_then(|v| v.as_u64()),
            Some(2)
        );
        assert_eq!(
            json.pointer("/excluded_tokens/left")
                .and_then(|v| v.as_u64()),
            Some(3)
        );
    }
}
