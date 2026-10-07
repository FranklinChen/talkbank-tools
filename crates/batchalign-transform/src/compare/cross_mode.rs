//! Typed morphotag and alignment comparisons for already-produced CHAT artifacts.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use serde::Serialize;
use talkbank_model::alignment::{
    WorSlotTiming, WorTimingBinding, WorTimingCorrespondence, corroborate_wor_timing,
};
use talkbank_model::model::{ChatFile, TranscriptName};
use talkbank_model::validation::{AlignmentValidation, ValidChatFile, ValidationPolicy};
use talkbank_model::{ErrorCollector, RuleSelection};
use talkbank_parser::TreeSitterParser;

use crate::{ValidatedParseError, parse_validated_with_parser};

mod morphotag;
pub use morphotag::{
    AnnotationPresence, MorphotagDifference, MorphotagPairResult, MorphotagTokenDifference,
    compare_validated_morphotag_plan,
};

use super::artifact::{ValidatedAlignmentPlan, ValidatedArtifactPair, ValidatedTranscriptionPlan};
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
    /// Retained CHAT failed full validation, including tier alignment.
    ArtifactInvalid { side: String, diagnostics: String },
    /// A producer failed; this is not a CHAT-invalidity verdict.
    ProducerFailure { side: String, detail: String },
    /// Speaker correspondence was absent or ambiguous.
    SpeakerCorrespondence { detail: String },
    /// An explicit speaker map named an absent speaker.
    InvalidSpeakerMap { detail: String },
    /// Token-level comparison cannot omit either document's speakers.
    IncompleteSpeakerMap {
        unmatched_left: Vec<String>,
        unmatched_right: Vec<String>,
    },
    /// Alignment requires identical normalized token identities.
    TokenIdentityMismatch {
        left_tokens: usize,
        right_tokens: usize,
    },
}

/// Timing representation for one side of an aligned token.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct TokenTiming {
    /// Start in milliseconds.
    pub start_ms: u64,
    /// End in milliseconds.
    pub end_ms: u64,
}

/// How far a `%wor` tier's slot count stands from the main tier's.
///
/// The fields are private and the only thing that fills them is
/// [`TokenTimingState::drifted`] in this module, reading chatter's own drift
/// payload. A row therefore cannot report a drift that nobody measured, and in
/// particular cannot be built by a consumer that did not look at the tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct WorSlotDrift {
    /// Slots the `%wor` tier actually carries.
    wor_slots: usize,
    /// Main-tier words the projection expected one slot each for.
    main_words: usize,
}

impl WorSlotDrift {
    /// Slots the `%wor` tier actually carries.
    pub fn wor_slots(&self) -> usize {
        self.wor_slots
    }

    /// Main-tier words the projection expected one slot each for.
    pub fn main_words(&self) -> usize {
        self.main_words
    }
}

/// How many `%wor` slots failed to corroborate the word they would time.
///
/// Private field, one constructor, for the reason [`WorSlotDrift`] gives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct LexicalMismatchCount {
    /// Slots whose display token did not match its main-tier word.
    mismatches: usize,
}

impl LexicalMismatchCount {
    /// Slots whose display token did not match its main-tier word.
    pub fn mismatches(&self) -> usize {
        self.mismatches
    }
}

/// Why one aligned token carries the timing it carries, or carries none.
///
/// Four of these five states were one bare `None` until 2026-09-16, and the
/// collapse answered four different questions with the same silence: a
/// corroborated slot that simply has no bullet, an utterance with no `%wor`
/// tier at all, a tier whose slot count has drifted from the main tier, and a
/// tier whose display tokens do not match the words they would time. A reader
/// of the report could see THAT a token had no timing and could not see which
/// of the four had happened, though chatter had said so in a payload this
/// module was discarding.
///
/// The three failure states are not interchangeable to anybody acting on them:
/// a missing tier means alignment never ran, a drifted one means the transcript
/// was edited after it ran, and an uncorroborated one means the tier belongs to
/// different words than the ones beside it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum TokenTimingState {
    /// A corroborated `%wor` tier times this slot.
    Timed(TokenTiming),
    /// A corroborated `%wor` tier carries no bullet for this slot. The tier is
    /// trustworthy and this word is simply not timed in it.
    Unaligned,
    /// The utterance carries no `%wor` tier, so no word in it is timed.
    NoWorTier,
    /// The `%wor` tier's slot count disagrees with the main tier's, so no slot
    /// can be paired with a word without guessing.
    WorTierDrifted(WorSlotDrift),
    /// Slot counts agreed, but display tokens did not match the words they
    /// would time, so the bullets belong to a different reading of the
    /// utterance.
    WorTierUncorroborated(LexicalMismatchCount),
}

impl TokenTimingState {
    /// Name a count drift, from chatter's own drift payload.
    ///
    /// Private, like the payload types' fields: the only caller is the one
    /// place that asked chatter and was told.
    fn drifted(wor_slots: usize, main_words: usize) -> Self {
        Self::WorTierDrifted(WorSlotDrift {
            wor_slots,
            main_words,
        })
    }

    /// Name a lexical corroboration failure, from chatter's own payload.
    fn uncorroborated(mismatches: usize) -> Self {
        Self::WorTierUncorroborated(LexicalMismatchCount { mismatches })
    }

    /// The timing, for the one state that has one.
    ///
    /// The other four arms are written out rather than swept into a `_`, so a
    /// sixth cause breaks this function instead of silently joining the
    /// untimed ones. This is the only place in the module that turns a state
    /// back into an `Option`, and it exists because a DELTA between two tokens
    /// genuinely requires both of them to be timed.
    pub fn timing(&self) -> Option<TokenTiming> {
        match self {
            Self::Timed(timing) => Some(*timing),
            Self::Unaligned
            | Self::NoWorTier
            | Self::WorTierDrifted(_)
            | Self::WorTierUncorroborated(_) => None,
        }
    }
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
    /// Left timing, or the named cause it has none.
    pub left_timing: TokenTimingState,
    /// Right timing, or the named cause it has none.
    pub right_timing: TokenTimingState,
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
            left.document(),
            right.document(),
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
    compare: impl Fn(&ValidChatFile, &ValidChatFile, &ValidatedArtifactPair) -> PairOutcome<T>,
) -> Vec<PairOutcome<T>> {
    let parser = match TreeSitterParser::new() {
        Ok(parser) => parser,
        Err(error) => {
            return pairs
                .iter()
                .map(|_| PairOutcome::Unpairable {
                    reason: PairFailureReason::ProducerFailure {
                        side: "both".to_string(),
                        detail: error.to_string(),
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
) -> Result<ValidChatFile, PairFailureReason> {
    let bytes = std::fs::read(path).map_err(|error| PairFailureReason::ArtifactRead {
        side: side.to_string(),
        detail: error.to_string(),
    })?;
    let text = String::from_utf8(bytes).map_err(|error| PairFailureReason::ArtifactRead {
        side: side.to_string(),
        detail: error.to_string(),
    })?;
    let errors = ErrorCollector::new();
    parse_validated_with_parser(
        parser,
        &text,
        ValidationPolicy::new(
            RuleSelection::new(),
            AlignmentValidation::IncludeTierAlignment,
        ),
        TranscriptName::for_path(path),
        &errors,
    )
    .map_err(|error| match error {
        ValidatedParseError::InternalFailure { failure, .. } => {
            PairFailureReason::ProducerFailure {
                side: side.to_string(),
                detail: failure.to_string(),
            }
        }
        ValidatedParseError::Validation(failure) if failure.has_internal_failure() => {
            PairFailureReason::ProducerFailure {
                side: side.to_string(),
                detail: failure.to_string(),
            }
        }
        ValidatedParseError::Parse(_) => PairFailureReason::ArtifactParse {
            side: side.to_string(),
            diagnostics: format!("{:?}", errors.into_vec()),
        },
        ValidatedParseError::Validation(failure) => PairFailureReason::ArtifactInvalid {
            side: side.to_string(),
            diagnostics: format!("{:?}", failure.diagnostics()),
        },
    })
}

/// A total speaker map bound to the admitted documents it was established for.
/// Neither token comparator can receive a merely injective, partial map.
struct CompleteSpeakerCorrespondence<'a> {
    left: &'a ValidChatFile,
    right: &'a ValidChatFile,
    map: SpeakerMap,
}

impl CompleteSpeakerCorrespondence<'_> {
    fn assignments(&self) -> &BTreeMap<String, String> {
        self.map.assignments()
    }
}

fn correspondence<'a>(
    left: &'a ValidChatFile,
    right: &'a ValidChatFile,
    pair: &ValidatedArtifactPair,
) -> Result<CompleteSpeakerCorrespondence<'a>, PairFailureReason> {
    let left_set: BTreeSet<String> = left
        .document()
        .unique_utterance_speakers()
        .into_iter()
        .map(|speaker| speaker.as_str().to_string())
        .collect();
    let right_set: BTreeSet<String> = right
        .document()
        .unique_utterance_speakers()
        .into_iter()
        .map(|speaker| speaker.as_str().to_string())
        .collect();
    let map = if let Some(assignments) = pair.speaker_map() {
        let map = SpeakerMap::try_from_assignments(assignments.clone()).map_err(|error| {
            PairFailureReason::InvalidSpeakerMap {
                detail: error.to_string(),
            }
        })?;
        for (left_speaker, right_speaker) in map.assignments() {
            if !left_set.contains(left_speaker) || !right_set.contains(right_speaker) {
                return Err(PairFailureReason::InvalidSpeakerMap {
                    detail: format!(
                        "mapped speaker {left_speaker:?} -> {right_speaker:?} is absent"
                    ),
                });
            }
        }
        map
    } else {
        match compare_transcripts_by_speaker(left.document(), right.document())
            .correspondence()
            .clone()
        {
            SpeakerCorrespondence::Established(map) => map,
            SpeakerCorrespondence::Unavailable => {
                return Err(PairFailureReason::SpeakerCorrespondence {
                    detail: "unavailable".to_string(),
                });
            }
            SpeakerCorrespondence::Ambiguous(candidates) => {
                return Err(PairFailureReason::SpeakerCorrespondence {
                    detail: format!("ambiguous ({} candidates)", candidates.candidates().len()),
                });
            }
        }
    };
    let assigned_left: BTreeSet<_> = map.assignments().keys().cloned().collect();
    let assigned_right: BTreeSet<_> = map.assignments().values().cloned().collect();
    let unmatched_left: Vec<_> = left_set.difference(&assigned_left).cloned().collect();
    let unmatched_right: Vec<_> = right_set.difference(&assigned_right).cloned().collect();
    if !unmatched_left.is_empty() || !unmatched_right.is_empty() {
        return Err(PairFailureReason::IncompleteSpeakerMap {
            unmatched_left,
            unmatched_right,
        });
    }
    Ok(CompleteSpeakerCorrespondence { left, right, map })
}

#[derive(Clone, PartialEq, Eq)]
struct AlignedToken {
    speaker: String,
    utterance: usize,
    token: usize,
    text: String,
    state: TokenTimingState,
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
        // another word's timing).
        //
        // Each way this can fail NAMES itself in the token's state, from the
        // payload chatter already hands back. Missing, drifted and
        // uncorroborated tiers all produced the same bare `None` until
        // 2026-09-16, indistinguishable from a corroborated slot that simply
        // carries no bullet, so the report could say a token had no timing but
        // never which of the four things had happened.
        let states: Vec<TokenTimingState> = match projection.bind_timing(utterance.wor_tier()) {
            WorTimingBinding::CountMatched(matched) => match corroborate_wor_timing(matched) {
                WorTimingCorrespondence::Corroborated(corroborated) => corroborated
                    .slots()
                    .iter()
                    .map(|slot| match slot.timing() {
                        WorSlotTiming::Timed(interval) => TokenTimingState::Timed(TokenTiming {
                            start_ms: interval.start().get(),
                            end_ms: interval.end().get(),
                        }),
                        WorSlotTiming::Unaligned => TokenTimingState::Unaligned,
                    })
                    .collect(),
                // The whole utterance shares one cause: corroboration is a
                // property of the tier, not of a slot.
                WorTimingCorrespondence::Uncorroborated(uncorroborated) => {
                    vec![
                        TokenTimingState::uncorroborated(uncorroborated.mismatches().len());
                        slot_texts.len()
                    ]
                }
            },
            WorTimingBinding::Missing(_) => vec![TokenTimingState::NoWorTier; slot_texts.len()],
            WorTimingBinding::Drifted(drift) => {
                vec![
                    TokenTimingState::drifted(drift.wor_count().get(), drift.main_count().get());
                    slot_texts.len()
                ]
            }
        };
        for (token, (text, state)) in slot_texts.into_iter().zip(states).enumerate() {
            result.push(AlignedToken {
                speaker: speaker.clone(),
                utterance: utterance_index,
                token,
                text,
                state,
            });
        }
    }
    result
}

fn compare_align_pair(
    left: &ValidChatFile,
    right: &ValidChatFile,
    pair: &ValidatedArtifactPair,
) -> PairOutcome<AlignmentPairResult> {
    let map = match correspondence(left, right, pair) {
        Ok(map) => map,
        Err(reason) => return PairOutcome::Unpairable { reason },
    };
    let left_tokens = alignment_tokens(map.left.document(), Some(map.assignments()));
    let right_tokens = alignment_tokens(map.right.document(), None);
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
        // A delta exists only where BOTH sides are timed, which is the one
        // question `timing()` exists to answer. Every other pairing, including
        // two tokens untimed for different reasons, has no delta to report and
        // now says why in the row itself.
        let paired = left.state.timing().zip(right.state.timing());
        let start_delta_ms = paired.map(|(l, r)| l.start_ms.abs_diff(r.start_ms));
        let end_delta_ms = paired.map(|(l, r)| l.end_ms.abs_diff(r.end_ms));
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
            left_timing: left.state,
            right_timing: right.state,
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
                    .state
                    .timing()
                    .zip(window[1].state.timing())
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
