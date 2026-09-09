//! The artifact this command produces: verdicts, with their provenance.
//!
//! # Why the provenance is in band
//!
//! An artifact records how it was made, beside itself, at the moment it is
//! made. Every number in this file depends on a choice somebody made: which
//! spans were enrolled, which threshold was stated, which acoustic model
//! produced the vectors, and which decode of the media they were computed
//! from. A reader six months later has none of that unless the file carries
//! it, and a reader who cannot reproduce a number treats it as unreproducible
//! rather than re-deriving it.
//!
//! # What this file is NOT
//!
//! It is not a corrected transcript, and it is not accuracy. A similarity is
//! AGREEMENT between two stretches of audio under one model; the enrollment
//! span is a human's claim about who is speaking and carries its own error
//! rate. The document says so in a field a reader cannot miss, because a bare
//! number between zero and one invites being read as a probability that the
//! speaker is who we say.

use serde::{Deserialize, Serialize};

use super::embedding::{EmbeddingDimension, MinimumEmbeddingFrames};
use super::enrollment::{EnrolledLabel, EnrollmentSet};
use super::model::EmbeddingModelRevision;
use super::policy::{LabelledScore, MatchThreshold, SpeakerVerdict, ThresholdPolicy};
use super::tracks::{PermutationPlan, TrackAnalysis, TrackContrast, TrackIdentity};
use crate::types::worker_v2::SpeakerEmbeddingBackendV2;

/// Schema version of the `<stem>_speaker_identity.json` artifact.
///
/// Bumped whenever a reader that understood the previous version would
/// misread this one. A new optional field is not a bump; a changed meaning is.
///
/// Version 2 (2026-09-09) added the track-level verdicts: `tracks`,
/// `track_contrasts` and `provenance.permutation`, all REQUIRED. A reader of
/// version 1 that ignored them would silently keep judging tracks by the
/// mean of their line scores, which is the verdict version 2 exists to
/// replace, so the change is a bump and not an optional field.
pub const SPEAKER_IDENTITY_SCHEMA_VERSION: u32 = 2;

/// The sentence every consumer has to have read.
///
/// A constant rather than prose in a book page, because the book page is not
/// in the file and the file is what gets forwarded.
const AGREEMENT_NOT_ACCURACY: &str = "Scores are acoustic AGREEMENT with an enrolled span under \
     one embedding model, not accuracy. The enrolled span is a human claim \
     about who is speaking and has its own error rate; no reference here is \
     gold.";

/// One enrolled voice, as it was given.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EnrolledSpanRecord {
    /// The name the caller gave this voice.
    pub label: EnrolledLabel,
    /// Start of the enrolled span, in file milliseconds.
    pub start_ms: u64,
    /// End of the enrolled span, in file milliseconds.
    pub end_ms: u64,
}

/// How this artifact was made.
///
/// No `Default`: every field is a fact only the run that produced the file
/// knows, and a defaulted one would be a fabricated provenance, which is worse
/// than an absent one because it looks like a record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeakerIdentityProvenance {
    /// Schema version of this document.
    pub schema_version: u32,
    /// The standing caveat, in the file rather than only in the book.
    pub interpretation: String,
    /// Transcript this run read, as the caller named it.
    pub transcript: String,
    /// Media the embeddings were computed from, as resolved.
    pub media: String,
    /// Sample rate of the single prepared decode every span indexes into.
    pub prepared_sample_rate_hz: u32,
    /// Embedding backend, as selected on the wire.
    pub embedding_backend: SpeakerEmbeddingBackendV2,
    /// Exact model revision the worker loaded, echoed from the worker.
    pub embedding_model_revision: EmbeddingModelRevision,
    /// Width of every vector this run compared, echoed from the worker.
    pub embedding_dimension: EmbeddingDimension,
    /// The model's own minimum span length, echoed from the worker.
    pub embedding_minimum_frames: MinimumEmbeddingFrames,
    /// The threshold the caller stated. There is no default; see
    /// [`super::policy`].
    pub match_threshold: MatchThreshold,
    /// The tiers whose utterances were scored.
    pub tiers: Vec<String>,
    /// Every enrolled span, in recording order.
    pub enrollments: Vec<EnrolledSpanRecord>,
    /// The seed and count behind every `track_contrasts` p-value.
    pub permutation: PermutationPlan,
    /// Build identity of the batchalign3 that wrote this.
    pub produced_by: String,
}

/// One utterance's verdict, with enough context to find it again.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UtteranceIdentity {
    /// Zero-based index of the utterance among the transcript's utterances.
    pub utterance_index: usize,
    /// One-based line number of its main tier in the transcript.
    pub line: usize,
    /// The speaker code the transcript currently carries.
    pub speaker: String,
    /// Start of its bullet, in file milliseconds, when it has one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start_ms: Option<u64>,
    /// End of its bullet, in file milliseconds, when it has one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_ms: Option<u64>,
    /// Similarity to EVERY enrolled voice, not only the best one.
    ///
    /// The full vector travels because a consumer choosing its own threshold
    /// has to be able to, without re-running the model. A file that carried
    /// only the winner would make every downstream question a new inference
    /// run.
    pub scores: Vec<LabelledScore>,
    /// What the stated policy concluded.
    pub verdict: SpeakerVerdict,
}

/// The whole artifact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeakerIdentityEvidence {
    /// How this was made.
    pub provenance: SpeakerIdentityProvenance,
    /// One entry per utterance of the named tiers, in transcript order.
    pub utterances: Vec<UtteranceIdentity>,
    /// One entry per speaker code among those utterances, as a voice.
    pub tracks: Vec<TrackIdentity>,
    /// One entry per enrolled voice: how far its best track stands out.
    pub track_contrasts: Vec<TrackContrast>,
}

/// Everything about a run that is not per-utterance, gathered so the document
/// cannot be assembled from half of it.
///
/// Taking this rather than eleven loose arguments is the difference between a
/// provenance block that is complete by construction and one that is complete
/// because the caller remembered. It is also what keeps the constructor below
/// under the workspace's wide-struct threshold at the call site.
#[derive(Debug, Clone, PartialEq)]
pub struct RunFacts {
    /// Transcript this run read, as the caller named it.
    pub transcript: String,
    /// Media the embeddings were computed from, as resolved.
    pub media: String,
    /// Sample rate of the single prepared decode.
    pub prepared_sample_rate_hz: u32,
    /// Embedding backend, as selected on the wire.
    pub embedding_backend: SpeakerEmbeddingBackendV2,
    /// Exact model revision the worker loaded.
    pub embedding_model_revision: EmbeddingModelRevision,
    /// The tiers whose utterances were scored.
    pub tiers: Vec<String>,
    /// Build identity of the batchalign3 that wrote this.
    pub produced_by: String,
}

/// Measurements learned only after a successful embedding response.
///
/// Kept separate from [`RunFacts`] so callers cannot fabricate placeholder
/// zeros before inference and rely on a later overwrite to make them true.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmbeddingRunFacts {
    /// Width shared by every returned vector.
    pub dimension: EmbeddingDimension,
    /// The model's smallest measurable span.
    pub minimum_frames: MinimumEmbeddingFrames,
}

impl SpeakerIdentityEvidence {
    /// Assemble the artifact from the run's own facts.
    ///
    /// The threshold is taken from the [`ThresholdPolicy`] that produced the
    /// verdicts rather than passed alongside them, so the number written into
    /// the file cannot disagree with the number the verdicts were decided
    /// under. Passing both would be two values held equal by convention, which
    /// is the shape where a wrong pairing type-checks.
    #[must_use]
    pub fn new(
        facts: RunFacts,
        embedding: EmbeddingRunFacts,
        enrollments: &EnrollmentSet,
        policy: &ThresholdPolicy,
        permutation: PermutationPlan,
        utterances: Vec<UtteranceIdentity>,
        tracks: TrackAnalysis,
    ) -> Self {
        Self {
            provenance: SpeakerIdentityProvenance {
                schema_version: SPEAKER_IDENTITY_SCHEMA_VERSION,
                interpretation: AGREEMENT_NOT_ACCURACY.to_owned(),
                transcript: facts.transcript,
                media: facts.media,
                prepared_sample_rate_hz: facts.prepared_sample_rate_hz,
                embedding_backend: facts.embedding_backend,
                embedding_model_revision: facts.embedding_model_revision,
                embedding_dimension: embedding.dimension,
                embedding_minimum_frames: embedding.minimum_frames,
                match_threshold: policy.threshold(),
                tiers: facts.tiers,
                enrollments: enrollments
                    .as_slice()
                    .iter()
                    .map(|enrollment| EnrolledSpanRecord {
                        label: enrollment.label().clone(),
                        start_ms: enrollment.window().start().get(),
                        end_ms: enrollment.window().end().get(),
                    })
                    .collect(),
                permutation,
                produced_by: facts.produced_by,
            },
            utterances,
            tracks: tracks.tracks,
            track_contrasts: tracks.contrasts,
        }
    }
}

#[cfg(test)]
#[allow(clippy::panic)]
mod tests {
    use super::super::enrollment::EnrollmentSpec;
    use super::super::policy::{SimilarityScore, UnscoredReason};
    use super::*;

    fn run_facts() -> RunFacts {
        RunFacts {
            transcript: "session.cha".to_owned(),
            media: "session.mp3".to_owned(),
            prepared_sample_rate_hz: 16_000,
            embedding_backend: SpeakerEmbeddingBackendV2::Pyannote,
            embedding_model_revision: super::super::pinned_embedding_revision()
                .expect("test: the packaged embedding manifest is valid"),
            tiers: vec!["*PAR0".to_owned()],
            produced_by: "batchalign3 test".to_owned(),
        }
    }

    fn evidence() -> SpeakerIdentityEvidence {
        let enrollments =
            match EnrollmentSet::new(vec![match EnrollmentSpec::parse("0-9000:INV") {
                Ok(spec) => spec,
                Err(error) => panic!("a legal enrollment: {error}"),
            }]) {
                Ok(set) => set,
                Err(error) => panic!("a legal set: {error}"),
            };
        let policy = ThresholdPolicy::new(match MatchThreshold::try_from(0.62) {
            Ok(threshold) => threshold,
            Err(error) => panic!("a legal threshold: {error}"),
        });
        let score = match SimilarityScore::try_from(0.81) {
            Ok(score) => score,
            Err(error) => panic!("a legal score: {error}"),
        };
        SpeakerIdentityEvidence::new(
            run_facts(),
            EmbeddingRunFacts {
                dimension: EmbeddingDimension::try_from(256)
                    .expect("test: a non-zero embedding dimension"),
                minimum_frames: MinimumEmbeddingFrames::try_from(1680)
                    .expect("test: a non-zero model minimum"),
            },
            &enrollments,
            &policy,
            PermutationPlan {
                seed: 0,
                count: super::super::tracks::PermutationCount::try_from(10)
                    .expect("test: a positive count"),
            },
            vec![
                UtteranceIdentity {
                    utterance_index: 0,
                    line: 12,
                    speaker: "PAR0".to_owned(),
                    start_ms: Some(10_000),
                    end_ms: Some(12_000),
                    scores: vec![LabelledScore {
                        label: match EnrolledLabel::parse("INV") {
                            Ok(label) => label,
                            Err(error) => panic!("a legal label: {error}"),
                        },
                        score,
                    }],
                    verdict: SpeakerVerdict::Matches {
                        label: match EnrolledLabel::parse("INV") {
                            Ok(label) => label,
                            Err(error) => panic!("a legal label: {error}"),
                        },
                        score,
                    },
                },
                UtteranceIdentity {
                    utterance_index: 1,
                    line: 14,
                    speaker: "PAR0".to_owned(),
                    start_ms: None,
                    end_ms: None,
                    scores: Vec::new(),
                    verdict: SpeakerVerdict::Unscored {
                        reason: UnscoredReason::NoBullet,
                    },
                },
            ],
            TrackAnalysis {
                tracks: Vec::new(),
                contrasts: Vec::new(),
            },
        )
    }

    /// WIRE FORMAT: the artifact roundtrips, and its provenance carries the
    /// threshold the verdicts were actually decided under.
    #[test]
    fn the_artifact_roundtrips_with_its_provenance() {
        let written = evidence();
        let json = match serde_json::to_string(&written) {
            Ok(json) => json,
            Err(error) => panic!("evidence serializes: {error}"),
        };
        let read: SpeakerIdentityEvidence = match serde_json::from_str(&json) {
            Ok(read) => read,
            Err(error) => panic!("evidence deserializes: {error}"),
        };
        assert_eq!(read, written);
        assert_eq!(
            read.provenance.schema_version,
            SPEAKER_IDENTITY_SCHEMA_VERSION
        );
        assert_eq!(read.provenance.match_threshold.get(), 0.62);
        assert_eq!(read.provenance.enrollments.len(), 1);
        assert_eq!(read.provenance.enrollments[0].label.as_str(), "INV");
    }

    /// The threshold in the provenance comes from the policy that produced the
    /// verdicts, so the file cannot state one number and have been decided
    /// under another.
    #[test]
    fn the_recorded_threshold_is_the_one_the_verdicts_used() {
        let value = match serde_json::to_value(evidence()) {
            Ok(value) => value,
            Err(error) => panic!("evidence serializes: {error}"),
        };
        assert_eq!(value["provenance"]["match_threshold"], 0.62);
    }

    /// The three verdict variants are distinguishable on the wire, and an
    /// unscored utterance carries a reason rather than a fabricated score.
    #[test]
    fn verdicts_are_tagged_and_unscored_carries_a_reason() {
        let value = match serde_json::to_value(evidence()) {
            Ok(value) => value,
            Err(error) => panic!("evidence serializes: {error}"),
        };
        assert_eq!(value["utterances"][0]["verdict"]["verdict"], "matches");
        assert_eq!(value["utterances"][1]["verdict"]["verdict"], "unscored");
        assert_eq!(value["utterances"][1]["verdict"]["reason"], "no_bullet");
        assert!(
            value["utterances"][1]["verdict"].get("score").is_none(),
            "an unscored utterance must carry no similarity at all"
        );
        assert!(
            value["utterances"][1].get("start_ms").is_none(),
            "an utterance with no bullet must carry no timing at all"
        );
    }

    /// The standing caveat travels IN the file, because the file is what gets
    /// forwarded and the book page is not.
    #[test]
    fn the_agreement_not_accuracy_caveat_is_in_the_document() {
        let value = match serde_json::to_value(evidence()) {
            Ok(value) => value,
            Err(error) => panic!("evidence serializes: {error}"),
        };
        let note = value["provenance"]["interpretation"]
            .as_str()
            .unwrap_or_default();
        assert!(note.contains("AGREEMENT"), "the caveat must be present");
        assert!(note.contains("not accuracy"));
    }
}
