//! Typed morphotag and alignment comparisons for already-produced CHAT artifacts.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use serde::Serialize;
use talkbank_model::ErrorCollector;
use talkbank_model::alignment::helpers::PositionalDomain;
use talkbank_model::alignment::{
    WorSlotTiming, WorTimingBinding, WorTimingCorrespondence, corroborate_wor_timing,
};
use talkbank_model::model::ChatFile;
use talkbank_parser::TreeSitterParser;

use crate::extract;

use super::artifact::{
    ValidatedAlignmentPlan, ValidatedArtifactPair, ValidatedMorphotagPlan,
    ValidatedTranscriptionPlan,
};
use super::cross_run::{
    CrossRunTranscriptionReport, SpeakerCorrespondence, SpeakerMap, compare_transcripts_by_speaker,
    compare_transcripts_with_exclusions,
};

/// A per-pair outcome. Ordinary differences are `Compared`; structural failures are typed.
#[allow(missing_docs)]
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum PairOutcome<T> {
    /// The pair was structurally comparable.
    Compared { result: T },
    /// The pair could not safely produce mode metrics.
    Unpairable { reason: PairFailureReason },
}

/// Structural reasons a pair could not be compared.
#[allow(missing_docs)]
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PairFailureReason {
    /// Artifact bytes could not be read as UTF-8.
    ArtifactRead { side: String, detail: String },
    /// CHAT parsing emitted diagnostics.
    ArtifactParse { side: String, diagnostics: String },
    /// Speaker correspondence was absent or ambiguous.
    SpeakerCorrespondence { detail: String },
    /// An explicit speaker map named an absent speaker.
    InvalidSpeakerMap { detail: String },
    /// Alignment requires identical normalized token identities.
    TokenIdentityMismatch {
        left_tokens: usize,
        right_tokens: usize,
    },
}

/// What differs between two morphotag analyses at one token.
///
/// A named set rather than seven parallel `bool` fields. The bools were
/// boolean blindness in its plainest form: nothing stopped a row asserting
/// every difference at once on a token that had none, `if row.lemma` read the
/// same as `if row.pos` at a glance, and adding an eighth axis meant editing
/// the struct, the constructor and the "is this row a difference" disjunction
/// in three separate places. As a set, "no differences" is `is_empty()`, the
/// axes are exhaustively `match`-able, and the wide-struct audit stops firing
/// because there is nothing wide left to audit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MorphotagDifference {
    /// The main-tier token text itself differs.
    Tokenization,
    /// Lemma differs.
    Lemma,
    /// Part of speech differs.
    Pos,
    /// Feature set differs (order-insensitive).
    FeatureSet,
    /// Clitic/chunk structure differs.
    CliticChunk,
    /// Mapped dependency-head token identity differs.
    DependencyHead,
    /// Dependency relation differs.
    Relation,
}

/// One structured morphotag difference at a main-tier token.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct MorphotagTokenDifference {
    /// Left speaker code.
    pub left_speaker: String,
    /// Right speaker code.
    pub right_speaker: String,
    /// Zero-based utterance ordinal within the mapped speaker.
    pub utterance: usize,
    /// Zero-based main-token position.
    pub token: usize,
    /// Normalized left main-tier token, if present.
    pub left_text: Option<String>,
    /// Normalized right main-tier token, if present.
    pub right_text: Option<String>,
    /// Every axis on which the two analyses disagree here.
    ///
    /// Empty means the two agree at this token, which is the state the seven
    /// bools could only express as "all of them are false".
    pub differences: std::collections::BTreeSet<MorphotagDifference>,
}

impl MorphotagTokenDifference {
    /// Whether the two analyses disagree at this token on any axis.
    pub fn differs(&self) -> bool {
        !self.differences.is_empty()
    }
}

/// Complete morphotag result for one artifact pair.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct MorphotagPairResult {
    /// One stable row per paired speaker/main-token position.
    pub tokens: Vec<MorphotagTokenDifference>,
    /// Difference rows only; identical tokens are intentionally omitted from review evidence.
    pub differences: Vec<MorphotagTokenDifference>,
    /// Number of aligned token positions examined.
    pub compared_tokens: usize,
}

/// Timing representation for one side of an aligned token.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct TokenTiming {
    /// Start in milliseconds.
    pub start_ms: u64,
    /// End in milliseconds.
    pub end_ms: u64,
}

/// One alignment timing comparison row.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AlignmentTokenDifference {
    /// Left speaker.
    pub left_speaker: String,
    /// Right speaker.
    pub right_speaker: String,
    /// Zero-based utterance ordinal within the mapped speaker.
    pub utterance: usize,
    /// Zero-based token position.
    pub token: usize,
    /// Normalized token identity.
    pub text: String,
    /// Left timing, explicitly absent when `%wor` has no bullet.
    pub left_timing: Option<TokenTiming>,
    /// Right timing, explicitly absent when `%wor` has no bullet.
    pub right_timing: Option<TokenTiming>,
    /// Absolute start delta when both timings exist.
    pub start_delta_ms: Option<u64>,
    /// Absolute end delta when both timings exist.
    pub end_delta_ms: Option<u64>,
}

/// Deterministic nearest-rank absolute-delta summary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TimingDistribution {
    /// Number of observed deltas.
    pub count: usize,
    /// Minimum delta.
    pub min_ms: Option<u64>,
    /// Median delta.
    pub median_ms: Option<u64>,
    /// 95th percentile by nearest rank.
    pub p95_ms: Option<u64>,
    /// Maximum delta.
    pub max_ms: Option<u64>,
}

/// Complete alignment result for one artifact pair.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AlignmentPairResult {
    /// One row per identical token identity.
    pub tokens: Vec<AlignmentTokenDifference>,
    /// Absolute start-delta distribution.
    pub start_deltas: TimingDistribution,
    /// Absolute end-delta distribution.
    pub end_deltas: TimingDistribution,
    /// Independent count of left-side timing order violations.
    pub left_order_violations: usize,
    /// Independent count of right-side timing order violations.
    pub right_order_violations: usize,
}

/// Compare every pair in a validated morphotag plan, continuing after pair failures.
pub fn compare_validated_morphotag_plan(
    plan: &ValidatedMorphotagPlan,
) -> Vec<PairOutcome<MorphotagPairResult>> {
    compare_pairs(plan.runs(), plan.artifact_pairs(), |left, right, pair| {
        compare_morph_pair(left, right, pair)
    })
}

/// Compare every pair in a validated alignment plan, continuing after pair failures.
pub fn compare_validated_alignment_plan(
    plan: &ValidatedAlignmentPlan,
) -> Vec<PairOutcome<AlignmentPairResult>> {
    compare_pairs(plan.runs(), plan.artifact_pairs(), |left, right, pair| {
        compare_align_pair(left, right, pair)
    })
}

/// Compare every pair in a validated transcription plan, continuing after pair failures.
pub fn compare_validated_transcription_pairs(
    plan: &ValidatedTranscriptionPlan,
) -> Vec<PairOutcome<CrossRunTranscriptionReport>> {
    compare_pairs(plan.runs(), plan.artifact_pairs(), |left, right, pair| {
        let explicit_map = match pair.speaker_map() {
            Some(assignments) => match SpeakerMap::try_from_assignments(assignments.clone()) {
                Ok(map) => Some(map),
                Err(error) => {
                    return PairOutcome::Unpairable {
                        reason: PairFailureReason::InvalidSpeakerMap {
                            detail: error.to_string(),
                        },
                    };
                }
            },
            None => None,
        };
        match compare_transcripts_with_exclusions(
            left,
            right,
            plan.exclusion_tokens(),
            explicit_map,
        ) {
            Ok(report) => PairOutcome::Compared { result: report },
            Err(error) => PairOutcome::Unpairable {
                reason: PairFailureReason::InvalidSpeakerMap {
                    detail: error.to_string(),
                },
            },
        }
    })
}

fn compare_pairs<T>(
    runs: &[super::artifact::ValidatedProducedRun; 2],
    pairs: &[ValidatedArtifactPair],
    compare: impl Fn(&ChatFile, &ChatFile, &ValidatedArtifactPair) -> PairOutcome<T>,
) -> Vec<PairOutcome<T>> {
    let parser = match TreeSitterParser::new() {
        Ok(parser) => parser,
        Err(error) => {
            return pairs
                .iter()
                .map(|_| PairOutcome::Unpairable {
                    reason: PairFailureReason::ArtifactParse {
                        side: "both".to_string(),
                        diagnostics: error.to_string(),
                    },
                })
                .collect();
        }
    };
    pairs
        .iter()
        .map(|pair| {
            let left_path = runs[0].artifacts().path().join(pair.left().as_str());
            let right_path = runs[1].artifacts().path().join(pair.right().as_str());
            let left = match parse(&parser, &left_path, "left") {
                Ok(file) => file,
                Err(reason) => return PairOutcome::Unpairable { reason },
            };
            let right = match parse(&parser, &right_path, "right") {
                Ok(file) => file,
                Err(reason) => return PairOutcome::Unpairable { reason },
            };
            compare(&left, &right, pair)
        })
        .collect()
}

fn parse(
    parser: &TreeSitterParser,
    path: &Path,
    side: &str,
) -> Result<ChatFile, PairFailureReason> {
    let bytes = std::fs::read(path).map_err(|error| PairFailureReason::ArtifactRead {
        side: side.to_string(),
        detail: error.to_string(),
    })?;
    let text = String::from_utf8(bytes).map_err(|error| PairFailureReason::ArtifactRead {
        side: side.to_string(),
        detail: error.to_string(),
    })?;
    let errors = ErrorCollector::new();
    let file = parser.parse_chat_file_streaming(&text, &errors);
    let diagnostics = errors.into_vec();
    if diagnostics.is_empty() {
        Ok(file)
    } else {
        Err(PairFailureReason::ArtifactParse {
            side: side.to_string(),
            diagnostics: format!("{diagnostics:?}"),
        })
    }
}

fn correspondence(
    left: &ChatFile,
    right: &ChatFile,
    pair: &ValidatedArtifactPair,
) -> Result<SpeakerMap, PairFailureReason> {
    if let Some(assignments) = pair.speaker_map() {
        let map = SpeakerMap::try_from_assignments(assignments.clone()).map_err(|error| {
            PairFailureReason::InvalidSpeakerMap {
                detail: error.to_string(),
            }
        })?;
        let left_set: BTreeSet<String> = left
            .unique_utterance_speakers()
            .into_iter()
            .map(|speaker| speaker.as_str().to_string())
            .collect();
        let right_set: BTreeSet<String> = right
            .unique_utterance_speakers()
            .into_iter()
            .map(|speaker| speaker.as_str().to_string())
            .collect();
        for (left_speaker, right_speaker) in map.assignments() {
            if !left_set.contains(left_speaker) || !right_set.contains(right_speaker) {
                return Err(PairFailureReason::InvalidSpeakerMap {
                    detail: format!(
                        "mapped speaker {left_speaker:?} -> {right_speaker:?} is absent"
                    ),
                });
            }
        }
        return Ok(map);
    }
    match compare_transcripts_by_speaker(left, right)
        .correspondence()
        .clone()
    {
        SpeakerCorrespondence::Established(map) => Ok(map),
        SpeakerCorrespondence::Unavailable => Err(PairFailureReason::SpeakerCorrespondence {
            detail: "unavailable".to_string(),
        }),
        SpeakerCorrespondence::Ambiguous(candidates) => {
            Err(PairFailureReason::SpeakerCorrespondence {
                detail: format!("ambiguous ({} candidates)", candidates.candidates().len()),
            })
        }
    }
}

#[derive(Clone)]
struct MorToken {
    text: String,
    lemma: Option<String>,
    pos: Option<String>,
    features: BTreeSet<String>,
    chunk_count: usize,
    head_identity: Option<String>,
    relation: Option<String>,
}

fn morph_tokens(file: &ChatFile) -> BTreeMap<String, Vec<Vec<MorToken>>> {
    let extracted = extract::extract_words(file, PositionalDomain::Mor);
    let utterances: Vec<_> = file.utterances().collect();
    let mut per_speaker: BTreeMap<String, Vec<Vec<MorToken>>> = BTreeMap::new();
    for entry in extracted {
        let Some(utterance) = utterances.get(entry.utterance_index.raw()) else {
            continue;
        };
        let mors = utterance.mor_tier().map(|tier| tier.items()).unwrap_or(&[]);
        let gras = utterance
            .gra_tier()
            .map(|tier| tier.relations())
            .unwrap_or(&[]);
        let mut chunk_starts = Vec::with_capacity(mors.len());
        let mut next_chunk = 0usize;
        for mor in mors {
            chunk_starts.push(next_chunk);
            next_chunk += mor.count_chunks();
        }
        let tokens = entry
            .words
            .iter()
            .enumerate()
            .map(|(index, word)| {
                let mor = mors.get(index);
                let relation = chunk_starts.get(index).and_then(|start| gras.get(*start));
                let head_identity = relation.and_then(|relation| {
                    if relation.head == 0 {
                        Some("ROOT".to_string())
                    } else {
                        chunk_starts
                            .iter()
                            .position(|start| *start + 1 == relation.head)
                            .map(|token| {
                                format!(
                                    "{}#{token}",
                                    word_identity(
                                        entry.speaker.as_str(),
                                        per_speaker.get(entry.speaker.as_str()).map_or(0, Vec::len)
                                    )
                                )
                            })
                    }
                });
                MorToken {
                    text: normalize(word.text.as_str()),
                    lemma: mor.map(|value| value.main.lemma.as_str().to_string()),
                    pos: mor.map(|value| value.main.pos.as_str().to_string()),
                    features: mor
                        .map(|value| {
                            value
                                .main
                                .features
                                .iter()
                                .map(ToString::to_string)
                                .collect()
                        })
                        .unwrap_or_default(),
                    chunk_count: mor.map_or(0, |value| value.count_chunks()),
                    head_identity,
                    relation: relation.map(|value| value.relation.as_str().to_string()),
                }
            })
            .collect();
        per_speaker
            .entry(entry.speaker.as_str().to_string())
            .or_default()
            .push(tokens);
    }
    per_speaker
}

fn compare_morph_pair(
    left: &ChatFile,
    right: &ChatFile,
    pair: &ValidatedArtifactPair,
) -> PairOutcome<MorphotagPairResult> {
    let map = match correspondence(left, right, pair) {
        Ok(map) => map,
        Err(reason) => return PairOutcome::Unpairable { reason },
    };
    let left_tokens = morph_tokens(left);
    let right_tokens = morph_tokens(right);
    let mut differences = Vec::new();
    let mut tokens = Vec::new();
    let mut compared_tokens = 0;
    for (left_speaker, right_speaker) in map.assignments() {
        let left_utts = left_tokens
            .get(left_speaker)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        let right_utts = right_tokens
            .get(right_speaker)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        for utterance in 0..left_utts.len().max(right_utts.len()) {
            let left_utt = left_utts.get(utterance).map(Vec::as_slice).unwrap_or(&[]);
            let right_utt = right_utts.get(utterance).map(Vec::as_slice).unwrap_or(&[]);
            for token in 0..left_utt.len().max(right_utt.len()) {
                compared_tokens += 1;
                let l = left_utt.get(token);
                let r = right_utt.get(token);
                let mut kinds = std::collections::BTreeSet::new();
                let mut note = |differs: bool, kind: MorphotagDifference| {
                    if differs {
                        kinds.insert(kind);
                    }
                };
                note(
                    l.map(|value| &value.text) != r.map(|value| &value.text),
                    MorphotagDifference::Tokenization,
                );
                note(
                    l.and_then(|value| value.lemma.as_ref())
                        != r.and_then(|value| value.lemma.as_ref()),
                    MorphotagDifference::Lemma,
                );
                note(
                    l.and_then(|value| value.pos.as_ref())
                        != r.and_then(|value| value.pos.as_ref()),
                    MorphotagDifference::Pos,
                );
                note(
                    l.map(|value| &value.features) != r.map(|value| &value.features),
                    MorphotagDifference::FeatureSet,
                );
                note(
                    l.map(|value| value.chunk_count) != r.map(|value| value.chunk_count),
                    MorphotagDifference::CliticChunk,
                );
                note(
                    l.and_then(|value| value.head_identity.as_ref())
                        != r.and_then(|value| value.head_identity.as_ref()),
                    MorphotagDifference::DependencyHead,
                );
                note(
                    l.and_then(|value| value.relation.as_ref())
                        != r.and_then(|value| value.relation.as_ref()),
                    MorphotagDifference::Relation,
                );

                let row = MorphotagTokenDifference {
                    left_speaker: left_speaker.clone(),
                    right_speaker: right_speaker.clone(),
                    utterance,
                    token,
                    left_text: l.map(|value| value.text.clone()),
                    right_text: r.map(|value| value.text.clone()),
                    differences: kinds,
                };
                if row.differs() {
                    differences.push(row.clone());
                }
                tokens.push(row);
            }
        }
    }
    PairOutcome::Compared {
        result: MorphotagPairResult {
            tokens,
            differences,
            compared_tokens,
        },
    }
}

#[derive(Clone, PartialEq, Eq)]
struct AlignedToken {
    speaker: String,
    utterance: usize,
    token: usize,
    text: String,
    timing: Option<TokenTiming>,
}

fn alignment_tokens(
    file: &ChatFile,
    speaker_map: Option<&BTreeMap<String, String>>,
) -> Vec<AlignedToken> {
    let mut speaker_ordinals = BTreeMap::<String, usize>::new();
    let mut result = Vec::new();
    for utterance in file.utterances() {
        let source_speaker = utterance.main.speaker.as_str().to_string();
        let speaker = speaker_map
            .and_then(|map| map.get(&source_speaker))
            .cloned()
            .unwrap_or(source_speaker.clone());
        let ordinal = speaker_ordinals.entry(speaker.clone()).or_default();
        let utterance_index = *ordinal;
        *ordinal += 1;
        // The `%wor` scale is the main tier's projection (chatter 0.23.0: the
        // count and the pairing are `WorMainTierProjection`'s, and the
        // extractor no longer offers a `%wor` domain). The projected tier
        // carries one word per slot; timing is read only through
        // `bind_timing`, which proves the `%wor` tier's slot count matches
        // before any pairing exists. A drifted or missing `%wor` tier yields
        // untimed tokens rather than the by-index zip this used to do, which
        // could pair a word with another word's bullet.
        let projection = utterance.main.wor_projection();
        let generated = projection.generate_tier();
        let slot_texts: Vec<String> = generated
            .words()
            .map(|word| normalize(word.cleaned_text()))
            .collect();
        // `generate_tier` borrows the projection and `bind_timing` consumes
        // it, so one projection (one walk of the main tier) serves both.
        // Timing is read only from a CORROBORATED pairing: equal counts admit
        // the positional pairing, and `corroborate_wor_timing` then proves
        // each `%wor` display token matches its main-tier word, which is what
        // chatter's type graph requires before a bullet may be trusted (a
        // same-count edit to either tier would otherwise pair a word with
        // another word's timing). Missing, drifted and uncorroborated tiers
        // all yield untimed tokens; which of the three it was is chatter's
        // diagnostic and is not carried into the comparison yet.
        let untimed = || vec![None; slot_texts.len()];
        let timings: Vec<Option<TokenTiming>> = match projection.bind_timing(utterance.wor_tier()) {
            WorTimingBinding::CountMatched(matched) => match corroborate_wor_timing(matched) {
                WorTimingCorrespondence::Corroborated(corroborated) => corroborated
                    .slots()
                    .iter()
                    .map(|slot| match slot.timing() {
                        WorSlotTiming::Timed(interval) => Some(TokenTiming {
                            start_ms: interval.start().get(),
                            end_ms: interval.end().get(),
                        }),
                        WorSlotTiming::Unaligned => None,
                    })
                    .collect(),
                WorTimingCorrespondence::Uncorroborated(_) => untimed(),
            },
            WorTimingBinding::Missing(_) | WorTimingBinding::Drifted(_) => untimed(),
        };
        for (token, (text, timing)) in slot_texts.into_iter().zip(timings).enumerate() {
            result.push(AlignedToken {
                speaker: speaker.clone(),
                utterance: utterance_index,
                token,
                text,
                timing,
            });
        }
    }
    result
}

fn compare_align_pair(
    left: &ChatFile,
    right: &ChatFile,
    pair: &ValidatedArtifactPair,
) -> PairOutcome<AlignmentPairResult> {
    let map = match correspondence(left, right, pair) {
        Ok(map) => map,
        Err(reason) => return PairOutcome::Unpairable { reason },
    };
    let left_tokens = alignment_tokens(left, Some(map.assignments()));
    let right_tokens = alignment_tokens(right, None);
    let left_identities: Vec<_> = left_tokens
        .iter()
        .map(|token| (&token.speaker, token.utterance, token.token, &token.text))
        .collect();
    let right_identities: Vec<_> = right_tokens
        .iter()
        .map(|token| (&token.speaker, token.utterance, token.token, &token.text))
        .collect();
    if left_identities != right_identities {
        return PairOutcome::Unpairable {
            reason: PairFailureReason::TokenIdentityMismatch {
                left_tokens: left_tokens.len(),
                right_tokens: right_tokens.len(),
            },
        };
    }
    let mut rows = Vec::with_capacity(left_tokens.len());
    let mut starts = Vec::new();
    let mut ends = Vec::new();
    for (left, right) in left_tokens.iter().zip(&right_tokens) {
        let start_delta_ms = left
            .timing
            .zip(right.timing)
            .map(|(l, r)| l.start_ms.abs_diff(r.start_ms));
        let end_delta_ms = left
            .timing
            .zip(right.timing)
            .map(|(l, r)| l.end_ms.abs_diff(r.end_ms));
        if let Some(value) = start_delta_ms {
            starts.push(value);
        }
        if let Some(value) = end_delta_ms {
            ends.push(value);
        }
        rows.push(AlignmentTokenDifference {
            left_speaker: map
                .assignments()
                .iter()
                .find_map(|(l, r)| (r == &left.speaker).then_some(l.clone()))
                .unwrap_or_else(|| left.speaker.clone()),
            right_speaker: right.speaker.clone(),
            utterance: left.utterance,
            token: left.token,
            text: left.text.clone(),
            left_timing: left.timing,
            right_timing: right.timing,
            start_delta_ms,
            end_delta_ms,
        });
    }
    PairOutcome::Compared {
        result: AlignmentPairResult {
            left_order_violations: order_violations(&left_tokens),
            right_order_violations: order_violations(&right_tokens),
            tokens: rows,
            start_deltas: distribution(starts),
            end_deltas: distribution(ends),
        },
    }
}

fn order_violations(tokens: &[AlignedToken]) -> usize {
    tokens
        .windows(2)
        .filter(|window| {
            window[0].speaker == window[1].speaker
                && window[0]
                    .timing
                    .zip(window[1].timing)
                    .is_some_and(|(left, right)| {
                        right.start_ms < left.start_ms || right.end_ms < left.end_ms
                    })
        })
        .count()
}

fn distribution(mut values: Vec<u64>) -> TimingDistribution {
    values.sort_unstable();
    let count = values.len();
    TimingDistribution {
        count,
        min_ms: values.first().copied(),
        median_ms: percentile(&values, 50),
        p95_ms: percentile(&values, 95),
        max_ms: values.last().copied(),
    }
}

fn percentile(values: &[u64], percentile: usize) -> Option<u64> {
    if values.is_empty() {
        None
    } else {
        Some(values[((values.len() * percentile).div_ceil(100)).saturating_sub(1)])
    }
}

fn normalize(value: &str) -> String {
    value.to_lowercase()
}
fn word_identity(speaker: &str, utterance: usize) -> String {
    format!("{speaker}:{utterance}")
}
