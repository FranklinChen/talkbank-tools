//! Similarity, the threshold policy, and the per-utterance verdict.
//!
//! # Why there is no default threshold anywhere in this file
//!
//! A threshold is a decision about how much acoustic agreement counts as the
//! same person, and it depends on the recording, the microphone, how much
//! enrollment audio there is and what the caller intends to do with a wrong
//! answer. Nothing in this crate knows any of that. A `Default` here would be
//! an invisible wrong value: every run would produce confident verdicts under
//! a number nobody chose and no output could show that it had been chosen by
//! accident. So [`MatchThreshold`] has no `Default`, [`ThresholdPolicy`] has
//! no `Default`, the CLI flag is required, and the chosen value is written
//! into the evidence beside the verdicts it produced.
//!
//! # The two decisions the policy makes, stated once
//!
//! 1. **A score exactly at the threshold matches.** The comparison is `>=`.
//!    `--threshold 0.5` means "0.5 or better". Both spellings are defensible;
//!    this one is the documented rule and the tests pin it as POLICY.
//! 2. **A tie at the maximum never matches.** The question the command
//!    answers is WHICH enrolled voice this is, and equal evidence for two of
//!    them does not answer it. Resolving the tie lexically would be
//!    deterministic and would still be a fact the evidence does not support.
//!    [`SpeakerVerdict::Matches`] therefore holds ONE label and
//!    [`SpeakerVerdict::NoMatch`] holds however many tied, so "matched two
//!    speakers" has no shape to travel through.

use std::fmt;

use serde::{Deserialize, Serialize};

/// A cosine similarity between two speaker embeddings.
///
/// The only producer is [`super::embedding::SpeakerEmbedding::similarity_to`].
/// There is no constructor from a bare `f64` other than the validating
/// [`TryFrom`], which exists for deserializing evidence written by an earlier
/// run and for tests. A value outside `[-1, 1]`, or a NaN, cannot be built at
/// all: NaN is the specific hazard, because the pinned embedding model returns
/// an all-NaN vector for input it cannot measure and a NaN compares false
/// against every threshold, so it would read as a considered "no match".
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct SimilarityScore(f64);

/// A similarity no cosine can produce.
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
#[error("{value} is not a cosine similarity: it must be a real number in [-1, 1]")]
pub struct NotASimilarity {
    /// The rejected value.
    pub value: f64,
}

impl SimilarityScore {
    /// The similarity as a number, for serialization and display.
    #[must_use]
    pub const fn get(self) -> f64 {
        self.0
    }
}

impl TryFrom<f64> for SimilarityScore {
    type Error = NotASimilarity;

    fn try_from(value: f64) -> Result<Self, Self::Error> {
        if value.is_finite() && (-1.0..=1.0).contains(&value) {
            Ok(Self(value))
        } else {
            Err(NotASimilarity { value })
        }
    }
}

impl<'de> Deserialize<'de> for SimilarityScore {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = f64::deserialize(deserializer)?;
        Self::try_from(value).map_err(serde::de::Error::custom)
    }
}

impl fmt::Display for SimilarityScore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.4}", self.0)
    }
}

/// The similarity at or above which an enrolled voice is called a match.
///
/// Same range as a similarity, and deliberately a DIFFERENT type: one is
/// measured and one is chosen, and a function taking both must not be able to
/// receive them in the wrong order.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct MatchThreshold(f64);

/// A threshold no similarity could ever reach or fall below.
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
#[error("{value} is not a usable match threshold: it must be a real number in [-1, 1]")]
pub struct NotAThreshold {
    /// The rejected value.
    pub value: f64,
}

impl MatchThreshold {
    /// The threshold as a number, for serialization into the evidence.
    #[must_use]
    pub const fn get(self) -> f64 {
        self.0
    }
}

impl TryFrom<f64> for MatchThreshold {
    type Error = NotAThreshold;

    fn try_from(value: f64) -> Result<Self, Self::Error> {
        if value.is_finite() && (-1.0..=1.0).contains(&value) {
            Ok(Self(value))
        } else {
            Err(NotAThreshold { value })
        }
    }
}

impl<'de> Deserialize<'de> for MatchThreshold {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = f64::deserialize(deserializer)?;
        Self::try_from(value).map_err(serde::de::Error::custom)
    }
}

impl fmt::Display for MatchThreshold {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.4}", self.0)
    }
}

/// One enrolled voice's similarity to one utterance.
///
/// The named pair is shared by the decision policy and the evidence writer, so
/// neither layer can fall back to positional tuples with an unchecked order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LabelledScore {
    /// The enrolled voice.
    pub label: super::enrollment::EnrolledLabel,
    /// Its similarity to this utterance.
    pub score: SimilarityScore,
}

/// Every enrolled label that achieved the highest score, and that score.
///
/// A list rather than one label, because the honest answer to a tie is "these
/// two, equally". Non-empty by construction: an enrollment set is non-empty,
/// so a scored utterance always has at least one best candidate, and an empty
/// `best` would be a state only a bug could produce.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BestScoringLabels {
    labels: Vec<super::enrollment::EnrolledLabel>,
    score: SimilarityScore,
}

/// A best-candidate list with nothing in it.
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
#[error("a scored utterance always has at least one best-scoring enrolled label")]
pub struct NoBestCandidate;

impl BestScoringLabels {
    /// The labels that tied at `score`, refusing an empty list.
    pub fn new(
        mut labels: Vec<super::enrollment::EnrolledLabel>,
        score: SimilarityScore,
    ) -> Result<Self, NoBestCandidate> {
        if labels.is_empty() {
            return Err(NoBestCandidate);
        }
        // Sorted so the evidence a reader compares across two runs does not
        // depend on the order enrollments happened to be given in.
        labels.sort();
        Ok(Self { labels, score })
    }

    /// The tied labels, in a stable order.
    #[must_use]
    pub fn labels(&self) -> Vec<&str> {
        self.labels.iter().map(|label| label.as_str()).collect()
    }

    /// The score they tied at.
    #[must_use]
    pub fn score(&self) -> SimilarityScore {
        self.score
    }
}

/// Why an utterance carries no similarity at all.
///
/// Variants rather than a string, because every one of these is acted on
/// differently by whoever reads the evidence: a too-short utterance might be
/// merged with its neighbour, a missing bullet is a transcript to time, audio
/// past the end of the recording is a media problem, and an utterance inside
/// the enrollment span is not evidence about anything because it IS the
/// evidence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "reason", rename_all = "snake_case")]
pub enum UnscoredReason {
    /// The utterance is shorter than the embedding model's own minimum.
    TooShortForEmbedding {
        /// How many prepared-audio frames the utterance spans.
        frames: u64,
        /// The model's own minimum, reported by the worker.
        minimum_frames: u64,
    },
    /// The utterance has no timing bullet, so there is no audio to embed.
    NoBullet,
    /// The main bullet was recovered as rejected syntax, not a numeric interval.
    UnparsedBullet,
    /// Every available enrollment vector was mathematically incomparable with
    /// this utterance, for example because one of the vectors had zero length.
    NoComparableEmbedding,
    /// The utterance's bullet names audio the recording does not contain.
    AudioMissing {
        /// Start of the bullet, in file milliseconds.
        start_ms: u64,
        /// End of the bullet, in file milliseconds.
        end_ms: u64,
        /// Length of the recording, in milliseconds.
        recording_ms: u64,
    },
    /// The utterance overlaps a span that was enrolled as a known voice.
    ///
    /// Scoring it would be circular: the enrolled vector was computed from
    /// this audio, so a high similarity says only that the arithmetic works.
    OverlapsEnrollment {
        /// The enrollment the utterance overlaps.
        label: super::enrollment::EnrolledLabel,
    },
}

impl UnscoredReason {
    /// A stable machine-readable code, for a consumer switching on the reason.
    ///
    /// Derived from the variant rather than written beside it, so a new
    /// variant cannot ship without one.
    #[must_use]
    pub const fn code(&self) -> &'static str {
        match self {
            Self::TooShortForEmbedding { .. } => "too_short_for_embedding",
            Self::NoBullet => "no_bullet",
            Self::UnparsedBullet => "unparsed_bullet",
            Self::NoComparableEmbedding => "no_comparable_embedding",
            Self::AudioMissing { .. } => "audio_missing",
            Self::OverlapsEnrollment { .. } => "overlaps_enrollment",
        }
    }
}

/// What this command concluded about one utterance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "verdict", rename_all = "snake_case")]
pub enum SpeakerVerdict {
    /// Exactly one enrolled voice reached the threshold, with nothing tied.
    Matches {
        /// The enrolled voice this utterance sounds like.
        label: super::enrollment::EnrolledLabel,
        /// Its similarity to this utterance.
        score: SimilarityScore,
    },
    /// No single enrolled voice was established.
    NoMatch {
        /// The best candidates and their shared score, so a reader can see
        /// how close the run came rather than only that it failed.
        best: BestScoringLabels,
    },
    /// No similarity was computed at all, and why.
    ///
    /// The reason is FLATTENED into the verdict object rather than nested
    /// under a key. Nesting an internally tagged enum inside a field of the
    /// same name produced `{"verdict": "unscored", "reason": {"reason":
    /// "no_bullet"}}`, which a reader has to look at twice to parse and which
    /// invites a consumer to read the wrong level.
    Unscored {
        /// The typed reason, flattened alongside `verdict`.
        #[serde(flatten)]
        reason: UnscoredReason,
    },
}

/// The caller's stated rule for turning similarities into verdicts.
///
/// Owns the threshold rather than taking it per call, so the whole run cannot
/// score half its utterances under one number and half under another.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThresholdPolicy {
    threshold: MatchThreshold,
}

impl ThresholdPolicy {
    /// The policy that calls a voice matched at or above `threshold`.
    ///
    /// There is no `Default` and no zero-argument constructor: see the module
    /// documentation for why the caller has to say.
    #[must_use]
    pub const fn new(threshold: MatchThreshold) -> Self {
        Self { threshold }
    }

    /// The threshold, for writing into the evidence beside the verdicts.
    #[must_use]
    pub const fn threshold(&self) -> MatchThreshold {
        self.threshold
    }

    /// Decide one utterance from its similarity to every enrolled voice.
    ///
    /// `scores` is non-empty in every production call, because an enrollment
    /// set is non-empty. The empty case is still handled rather than
    /// panicking, and it reports the condition it actually is: nothing was
    /// scored.
    #[must_use]
    pub fn verdict(&self, scores: &[LabelledScore]) -> SpeakerVerdict {
        let Some(best_score) = scores
            .iter()
            .map(|labelled| labelled.score)
            .max_by(|left, right| left.get().total_cmp(&right.get()))
        else {
            return SpeakerVerdict::Unscored {
                reason: UnscoredReason::NoComparableEmbedding,
            };
        };

        let tied: Vec<super::enrollment::EnrolledLabel> = scores
            .iter()
            .filter(|labelled| labelled.score == best_score)
            .map(|labelled| labelled.label.clone())
            .collect();

        match (tied.len(), best_score.get() >= self.threshold.get()) {
            // The only route to a match: one label, and it cleared the bar.
            (1, true) => match tied.into_iter().next() {
                Some(label) => SpeakerVerdict::Matches {
                    label,
                    score: best_score,
                },
                // Unreachable by the arm's own guard, and written out rather
                // than unwrapped so no panic exists in this path at all.
                None => SpeakerVerdict::Unscored {
                    reason: UnscoredReason::NoComparableEmbedding,
                },
            },
            // Below the bar, or tied. Both report the best candidates; a tie
            // above the bar is deliberately not a match (see the module docs).
            _ => match BestScoringLabels::new(tied, best_score) {
                Ok(best) => SpeakerVerdict::NoMatch { best },
                Err(_) => SpeakerVerdict::Unscored {
                    reason: UnscoredReason::NoComparableEmbedding,
                },
            },
        }
    }
}

#[cfg(test)]
#[allow(clippy::panic)]
mod tests {
    use super::super::EnrolledLabel;
    use super::*;

    fn label(name: &str) -> EnrolledLabel {
        match EnrolledLabel::parse(name) {
            Ok(label) => label,
            Err(error) => panic!("a legal label: {error}"),
        }
    }

    fn score(value: f64) -> SimilarityScore {
        match SimilarityScore::try_from(value) {
            Ok(score) => score,
            Err(error) => panic!("a legal similarity: {error}"),
        }
    }

    fn policy(threshold: f64) -> ThresholdPolicy {
        match MatchThreshold::try_from(threshold) {
            Ok(threshold) => ThresholdPolicy::new(threshold),
            Err(error) => panic!("a legal threshold: {error}"),
        }
    }

    fn labelled(name: &str, value: f64) -> LabelledScore {
        LabelledScore {
            label: label(name),
            score: score(value),
        }
    }

    /// POLICY: a score EXACTLY at the threshold matches.
    ///
    /// A policy test, not an invariant: `>=` and `>` are both defensible, and
    /// the choice has to be written somewhere a reader can find it.
    #[test]
    fn a_score_exactly_at_the_threshold_matches() {
        match policy(0.5).verdict(&[labelled("INV", 0.5)]) {
            SpeakerVerdict::Matches { label, score } => {
                assert_eq!(label.as_str(), "INV");
                assert_eq!(score.get(), 0.5);
            }
            other => panic!("expected a match at the threshold, got {other:?}"),
        }
    }

    /// POLICY: below the threshold there is no match, and the verdict still
    /// reports what the best candidate was and how close it came.
    #[test]
    fn just_below_the_threshold_reports_the_best_candidate() {
        match policy(0.5).verdict(&[labelled("INV", 0.49), labelled("CHI", 0.10)]) {
            SpeakerVerdict::NoMatch { best } => {
                assert_eq!(best.labels(), ["INV"]);
                assert_eq!(best.score().get(), 0.49);
            }
            other => panic!("expected no match below the threshold, got {other:?}"),
        }
    }

    /// POLICY: two enrolled voices tying at the maximum never match, even when
    /// the tied score clears the threshold.
    #[test]
    fn a_tie_at_or_above_the_threshold_is_not_a_match() {
        match policy(0.5).verdict(&[labelled("INV", 0.90), labelled("CHI", 0.90)]) {
            SpeakerVerdict::NoMatch { best } => {
                assert_eq!(best.labels(), ["CHI", "INV"]);
                assert_eq!(best.score().get(), 0.90);
            }
            other => panic!("a tie must not resolve to a speaker, got {other:?}"),
        }
    }

    /// A match carries exactly one label, so "matched two speakers" is not a
    /// value this type can hold. The test records the property; the variant's
    /// shape is what enforces it.
    #[test]
    fn a_match_carries_exactly_one_label_by_construction() {
        assert!(matches!(
            policy(0.0).verdict(&[labelled("INV", 1.0)]),
            SpeakerVerdict::Matches { .. }
        ));
    }

    /// An unmeasurable utterance is UNSCORED under a typed reason, never a
    /// fabricated zero similarity, and the reason's code comes from its own
    /// variant so a new reason cannot ship without one.
    #[test]
    fn an_unmeasurable_utterance_is_unscored_with_a_typed_reason() {
        let reason = UnscoredReason::TooShortForEmbedding {
            frames: 400,
            minimum_frames: 1680,
        };
        assert_eq!(reason.code(), "too_short_for_embedding");
        assert_eq!(UnscoredReason::NoBullet.code(), "no_bullet");
        assert_eq!(
            policy(0.0).verdict(&[]),
            SpeakerVerdict::Unscored {
                reason: UnscoredReason::NoComparableEmbedding,
            }
        );
        assert!(matches!(
            SpeakerVerdict::Unscored { reason },
            SpeakerVerdict::Unscored { .. }
        ));
    }

    /// A similarity outside the cosine range, or a NaN, is refused at
    /// construction, so a value only a bug could produce never reaches a
    /// comparison that would silently read it as "no match".
    #[test]
    fn a_similarity_outside_the_cosine_range_is_refused() {
        assert!(SimilarityScore::try_from(1.000_001).is_err());
        assert!(SimilarityScore::try_from(-1.000_001).is_err());
        assert!(SimilarityScore::try_from(f64::NAN).is_err());
        assert!(SimilarityScore::try_from(f64::INFINITY).is_err());
        assert!(SimilarityScore::try_from(1.0).is_ok());
    }

    /// WIRE FORMAT: a similarity read back from an evidence file goes through
    /// the same refusal as one computed in-process.
    #[test]
    fn a_deserialized_similarity_is_validated_too() {
        assert!(serde_json::from_str::<SimilarityScore>("2.0").is_err());
        assert!(serde_json::from_str::<SimilarityScore>("0.5").is_ok());
    }

    #[test]
    fn a_threshold_no_similarity_can_reach_is_refused() {
        assert!(MatchThreshold::try_from(1.5).is_err());
        assert!(MatchThreshold::try_from(f64::NAN).is_err());
        assert!(MatchThreshold::try_from(-1.0).is_ok());
    }

    #[test]
    fn best_scoring_labels_cannot_be_empty() {
        assert!(BestScoringLabels::new(Vec::new(), score(0.1)).is_err());
    }
}
