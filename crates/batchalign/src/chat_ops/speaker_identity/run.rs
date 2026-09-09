//! The whole decision, as one function over an injected inference capability.
//!
//! # Why the model is a trait and not a call
//!
//! Everything interesting about this command is arithmetic and policy over
//! vectors: which spans to ask about, what to do when a span cannot be
//! measured, and how a set of similarities becomes a verdict. None of that
//! needs a model to test, and all of it is where a defect would be invisible.
//! So [`SpeakerEmbeddingInference`] is a capability the caller supplies, and
//! every decision below is provable with a fake that returns chosen vectors.
//!
//! This is dependency injection without an authorization escape hatch, in the
//! same shape the Rev evidence boundary uses: the trait method consumes an
//! [`EmbeddingRequest`], which only [`identify_speakers`] builds, so a fake can
//! prove the pipeline end to end while production still cannot ask the worker
//! for a span that never went through [`PreparedPcm::locate`].

use std::collections::BTreeMap;

use async_trait::async_trait;

use crate::media::window::MediaWindow;
use crate::time::FileMs;

use super::embedding::{EmbeddingDimension, MinimumEmbeddingFrames, SpeakerEmbedding};
use super::enrollment::{EnrolledLabel, EnrollmentSet};
use super::evidence::{EmbeddingRunFacts, RunFacts, SpeakerIdentityEvidence, UtteranceIdentity};
use super::frames::{FrameSpan, OutsidePreparedAudio, PreparedPcm};
use super::policy::{LabelledScore, SpeakerVerdict, ThresholdPolicy, UnscoredReason};
use super::tracks::{
    EmptyTrackCode, EnrolledVoice, PermutationPlan, TrackAnalysisFailure, TrackCode, TrackLines,
};

/// One timed, or untimed, utterance of a named tier, as read from the model.
///
/// Read from the typed CHAT AST by the caller. This module never parses CHAT:
/// a bullet arrives here already as [`FileMs`], the space it is written in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranscriptUtterance {
    /// Zero-based index among the transcript's utterances.
    pub utterance_index: usize,
    /// One-based line number of its main tier.
    pub line: usize,
    /// The speaker code the transcript carries.
    pub speaker: String,
    /// The timing state read from its bullet.
    pub timing: UtteranceTiming,
}

/// Timing evidence carried by one transcript utterance.
///
/// Parsing a bullet and proving it is a non-empty media window happen where
/// the utterance is read. Downstream code therefore cannot accidentally treat
/// a reversed or zero-length pair as a window that may be embedded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UtteranceTiming {
    /// The main tier carries no timing bullet.
    Untimed,
    /// The bullet exists but cannot describe a non-empty audio window.
    Invalid {
        /// Start written in the bullet.
        start: FileMs,
        /// End written in the bullet.
        end: FileMs,
    },
    /// A non-empty window ready for containment checking against the decode.
    Window(MediaWindow),
}

impl UtteranceTiming {
    fn start(self) -> Option<FileMs> {
        match self {
            Self::Untimed => None,
            Self::Invalid { start, .. } => Some(start),
            Self::Window(window) => Some(window.start()),
        }
    }

    fn end(self) -> Option<FileMs> {
        match self {
            Self::Untimed => None,
            Self::Invalid { end, .. } => Some(end),
            Self::Window(window) => Some(window.end()),
        }
    }
}

/// One span the worker is being asked about, named.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestedSpan {
    /// The name the response must echo.
    pub span_id: String,
    /// Where in the prepared decode it lives.
    pub frames: FrameSpan,
}

/// Every span one run asks about, in one request.
///
/// Built only by [`identify_speakers`], which is what stops a caller asking
/// the worker about frames that never went through [`PreparedPcm::locate`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbeddingRequest {
    spans: Vec<RequestedSpan>,
}

impl EmbeddingRequest {
    /// The spans to embed.
    #[must_use]
    pub fn spans(&self) -> &[RequestedSpan] {
        &self.spans
    }

    /// Build a request fixture after tests have created located spans through
    /// the same [`PreparedPcm`] transition production uses.
    #[cfg(test)]
    pub(crate) fn for_test(spans: Vec<RequestedSpan>) -> Self {
        Self { spans }
    }
}

/// What the model said about one span.
#[derive(Debug, Clone, PartialEq)]
pub enum SpanOutcome {
    /// Measured.
    Embedded(SpeakerEmbedding),
    /// Below the model's own minimum length.
    TooShort {
        /// How many frames the span held.
        frames: u64,
    },
}

/// Everything one embedding run returns, including the model's own bounds.
#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddingResponse {
    /// One outcome per requested span, by span id.
    pub outcomes: BTreeMap<String, SpanOutcome>,
    /// The model's own minimum span length, for reporting a refusal.
    pub minimum_frames: MinimumEmbeddingFrames,
    /// Width of every vector, for the evidence's provenance.
    pub dimension: EmbeddingDimension,
}

/// Failure of the injected embedding capability.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum EmbeddingInferenceFailure {
    /// The worker request could not be completed.
    #[error("worker dispatch failed: {detail}")]
    Dispatch {
        /// The transport or worker detail.
        detail: String,
    },
    /// The worker answered, but its payload did not satisfy the embedding
    /// response contract.
    #[error("worker returned an invalid embedding response: {detail}")]
    InvalidResponse {
        /// The response-contract detail.
        detail: String,
    },
}

/// The capability that turns spans into vectors.
#[async_trait]
pub trait SpeakerEmbeddingInference: Send + Sync {
    /// Embed every span of one prepared decode.
    async fn embed(
        &self,
        request: EmbeddingRequest,
    ) -> Result<EmbeddingResponse, EmbeddingInferenceFailure>;
}

/// Why a run could not produce evidence at all.
///
/// Distinct from a per-utterance [`UnscoredReason`]: these end the run, and
/// none of them is a fact about one utterance.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum SpeakerIdentityFailure {
    /// An enrolled span names audio the recording does not contain.
    ///
    /// Fatal rather than per-utterance: an enrollment is the caller's claim
    /// about which voice is which, and continuing without one would silently
    /// answer a different question from the one that was asked.
    #[error("the enrolled span {label} is not inside the recording: {source}")]
    EnrollmentOutsideRecording {
        /// The enrollment that could not be located.
        label: EnrolledLabel,
        /// The typed containment failure.
        source: OutsidePreparedAudio,
    },
    /// An enrolled span is shorter than the model can measure.
    #[error(
        "the enrolled span {label} is {frames} frames, and the model needs at least \
         {minimum_frames}; enroll a longer stretch of audio"
    )]
    EnrollmentTooShort {
        /// The enrollment that could not be measured.
        label: EnrolledLabel,
        /// Its length in prepared-audio frames.
        frames: u64,
        /// The model's minimum.
        minimum_frames: u64,
    },
    /// The worker did not answer about a span that was requested.
    #[error("the worker returned no outcome for the span {span_id}")]
    MissingOutcome {
        /// The span with no answer.
        span_id: String,
    },
    /// The inference capability itself failed.
    #[error(transparent)]
    Inference(#[from] EmbeddingInferenceFailure),
    /// A scored utterance carries an empty speaker code, which no main tier
    /// can have; the transcript reader has handed over something it did not
    /// parse.
    #[error("the utterance on line {line} has an empty speaker code: {source}")]
    EmptySpeakerCode {
        /// One-based line of the main tier.
        line: usize,
        /// Why.
        #[source]
        source: EmptyTrackCode,
    },
    /// The tracks could not be scored as voices.
    #[error(transparent)]
    Tracks(#[from] TrackAnalysisFailure),
}

/// Span id for one enrollment. Prefixed so it can never collide with an
/// utterance id, which is what a bare label would risk.
fn enrollment_span_id(label: &EnrolledLabel) -> String {
    format!("enroll:{label}")
}

/// Span id for one utterance.
fn utterance_span_id(index: usize) -> String {
    format!("utt:{index}")
}

/// Score every timed utterance against every enrolled voice.
///
/// The order of operations is the type graph in the module documentation, and
/// each step's failure is reported as the thing it is: an enrollment that
/// cannot be measured ends the run, an utterance that cannot be measured is
/// one UNSCORED verdict.
pub async fn identify_speakers(
    facts: RunFacts,
    enrollments: &EnrollmentSet,
    utterances: &[TranscriptUtterance],
    prepared: PreparedPcm,
    policy: &ThresholdPolicy,
    permutation: PermutationPlan,
    inference: &dyn SpeakerEmbeddingInference,
) -> Result<SpeakerIdentityEvidence, SpeakerIdentityFailure> {
    let mut spans = Vec::new();

    for enrollment in enrollments.as_slice() {
        let frames = prepared.locate(enrollment.window()).map_err(|error| {
            SpeakerIdentityFailure::EnrollmentOutsideRecording {
                label: enrollment.label().clone(),
                source: error,
            }
        })?;
        spans.push(RequestedSpan {
            span_id: enrollment_span_id(enrollment.label()),
            frames,
        });
    }

    // An utterance gets a span only when it has a bullet, that bullet is a
    // non-empty window, it lies inside the prepared decode, and it is not
    // itself enrollment material. Each of those failures is recorded HERE, as
    // the verdict it produces, rather than being discovered later by something
    // that has forgotten which utterance it was looking at.
    let mut refusals: BTreeMap<usize, UnscoredReason> = BTreeMap::new();
    for utterance in utterances {
        let window = match utterance.timing {
            UtteranceTiming::Untimed => {
                refusals.insert(utterance.utterance_index, UnscoredReason::NoBullet);
                continue;
            }
            UtteranceTiming::Invalid { start, end } => {
                refusals.insert(
                    utterance.utterance_index,
                    UnscoredReason::AudioMissing {
                        start_ms: start.get(),
                        end_ms: end.get(),
                        recording_ms: prepared.duration().0,
                    },
                );
                continue;
            }
            UtteranceTiming::Window(window) => window,
        };
        if let Some(label) = enrollments.overlapping(window) {
            refusals.insert(
                utterance.utterance_index,
                UnscoredReason::OverlapsEnrollment {
                    label: label.clone(),
                },
            );
            continue;
        }
        match prepared.locate(window) {
            Ok(frames) => spans.push(RequestedSpan {
                span_id: utterance_span_id(utterance.utterance_index),
                frames,
            }),
            Err(outside) => {
                refusals.insert(
                    utterance.utterance_index,
                    UnscoredReason::AudioMissing {
                        start_ms: outside.start_ms,
                        end_ms: outside.end_ms,
                        recording_ms: outside.recording_ms,
                    },
                );
            }
        }
    }

    let response = inference.embed(EmbeddingRequest { spans }).await?;

    let mut enrolled: Vec<EnrolledVoice> = Vec::new();
    for enrollment in enrollments.as_slice() {
        let span_id = enrollment_span_id(enrollment.label());
        match response.outcomes.get(&span_id) {
            Some(SpanOutcome::Embedded(vector)) => {
                enrolled.push(EnrolledVoice {
                    label: enrollment.label().clone(),
                    vector: vector.clone(),
                });
            }
            Some(SpanOutcome::TooShort { frames }) => {
                return Err(SpeakerIdentityFailure::EnrollmentTooShort {
                    label: enrollment.label().clone(),
                    frames: *frames,
                    minimum_frames: response.minimum_frames.get(),
                });
            }
            None => {
                return Err(SpeakerIdentityFailure::MissingOutcome { span_id });
            }
        }
    }

    let mut identified = Vec::with_capacity(utterances.len());
    // Every line lands on its track as it is judged, embedded or refused, so
    // the track analysis below sees exactly the population the line verdicts
    // saw and neither can drift from the other.
    let mut track_lines = TrackLines::default();
    for utterance in utterances {
        let start_ms = utterance.timing.start().map(FileMs::get);
        let end_ms = utterance.timing.end().map(FileMs::get);
        let track = TrackCode::from_speaker(&utterance.speaker).map_err(|source| {
            SpeakerIdentityFailure::EmptySpeakerCode {
                line: utterance.line,
                source,
            }
        })?;

        let verdict_and_scores = match refusals.get(&utterance.utterance_index) {
            Some(reason) => {
                track_lines.refused(track);
                (
                    SpeakerVerdict::Unscored {
                        reason: reason.clone(),
                    },
                    Vec::new(),
                )
            }
            None => {
                let span_id = utterance_span_id(utterance.utterance_index);
                match response.outcomes.get(&span_id) {
                    Some(SpanOutcome::TooShort { frames }) => {
                        track_lines.refused(track);
                        (
                            SpeakerVerdict::Unscored {
                                reason: UnscoredReason::TooShortForEmbedding {
                                    frames: *frames,
                                    minimum_frames: response.minimum_frames.get(),
                                },
                            },
                            Vec::new(),
                        )
                    }
                    Some(SpanOutcome::Embedded(vector)) => {
                        let mut scored = Vec::with_capacity(enrolled.len());
                        for voice in &enrolled {
                            match voice.vector.similarity_to(vector) {
                                Ok(score) => scored.push(LabelledScore {
                                    label: voice.label.clone(),
                                    score,
                                }),
                                // An incomparable pair means one of the two
                                // vectors has no direction or came from
                                // another model. Dropping it from the score
                                // list is what makes the verdict honest: a
                                // fabricated similarity would be indis-
                                // tinguishable from a measured one.
                                Err(_) => continue,
                            }
                        }
                        // A line joins its track's voice only once it has
                        // compared with at least one enrolled voice: a
                        // directionless vector is one unscored LINE, as it
                        // always was, and not a reason to refuse the whole
                        // session's tracks.
                        if scored.is_empty() {
                            track_lines.refused(track);
                        } else {
                            track_lines.embedded(track, vector.clone());
                        }
                        let verdict = policy.verdict(&scored);
                        (verdict, scored)
                    }
                    None => {
                        return Err(SpeakerIdentityFailure::MissingOutcome { span_id });
                    }
                }
            }
        };

        let (verdict, scores) = verdict_and_scores;
        identified.push(UtteranceIdentity {
            utterance_index: utterance.utterance_index,
            line: utterance.line,
            speaker: utterance.speaker.clone(),
            start_ms,
            end_ms,
            scores,
            verdict,
        });
    }

    let tracks = track_lines.analyse(&enrolled, permutation)?;

    Ok(SpeakerIdentityEvidence::new(
        facts,
        EmbeddingRunFacts {
            dimension: response.dimension,
            minimum_frames: response.minimum_frames,
        },
        enrollments,
        policy,
        permutation,
        identified,
        tracks,
    ))
}

#[cfg(test)]
#[allow(clippy::panic)]
mod tests {
    use super::super::enrollment::EnrollmentSpec;
    use super::super::model::pinned_embedding_revision;
    use super::super::policy::MatchThreshold;
    use super::super::tracks::{TrackContrast, TrackIdentity, documented_permutation_plan};
    use super::*;

    /// A model that answers with vectors the test chose, so every decision
    /// below is provable without loading anything.
    struct ChosenVectors {
        by_span: BTreeMap<String, SpanOutcome>,
        minimum_frames: u64,
    }

    #[async_trait]
    impl SpeakerEmbeddingInference for ChosenVectors {
        async fn embed(
            &self,
            request: EmbeddingRequest,
        ) -> Result<EmbeddingResponse, EmbeddingInferenceFailure> {
            let mut outcomes = BTreeMap::new();
            for span in request.spans() {
                if let Some(outcome) = self.by_span.get(&span.span_id) {
                    outcomes.insert(span.span_id.clone(), outcome.clone());
                }
            }
            Ok(EmbeddingResponse {
                outcomes,
                minimum_frames: MinimumEmbeddingFrames::try_from(self.minimum_frames)
                    .expect("test: a non-zero model minimum"),
                dimension: EmbeddingDimension::try_from(2)
                    .expect("test: a non-zero embedding dimension"),
            })
        }
    }

    fn vector(components: Vec<f64>) -> SpanOutcome {
        let width = components.len();
        match SpeakerEmbedding::from_worker(components, width) {
            Ok(value) => SpanOutcome::Embedded(value),
            Err(error) => panic!("a legal embedding: {error}"),
        }
    }

    fn enrollments(specs: &[&str]) -> EnrollmentSet {
        let parsed = specs
            .iter()
            .map(|spec| match EnrollmentSpec::parse(spec) {
                Ok(spec) => spec,
                Err(error) => panic!("a legal enrollment: {error}"),
            })
            .collect();
        match EnrollmentSet::new(parsed) {
            Ok(set) => set,
            Err(error) => panic!("a legal set: {error}"),
        }
    }

    fn prepared() -> PreparedPcm {
        match PreparedPcm::new(16_000, 16_000 * 60) {
            Ok(pcm) => pcm,
            Err(error) => panic!("a legal decode: {error}"),
        }
    }

    fn policy(threshold: f64) -> ThresholdPolicy {
        match MatchThreshold::try_from(threshold) {
            Ok(threshold) => ThresholdPolicy::new(threshold),
            Err(error) => panic!("a legal threshold: {error}"),
        }
    }

    fn facts() -> RunFacts {
        RunFacts {
            transcript: "session.cha".to_owned(),
            media: "session.mp3".to_owned(),
            prepared_sample_rate_hz: 16_000,
            embedding_backend: crate::types::worker_v2::SpeakerEmbeddingBackendV2::Pyannote,
            embedding_model_revision: pinned_embedding_revision()
                .expect("test: the packaged embedding manifest is valid"),
            tiers: vec!["*PAR0".to_owned()],
            produced_by: "test".to_owned(),
        }
    }

    fn timed(index: usize, start_ms: u64, end_ms: u64) -> TranscriptUtterance {
        TranscriptUtterance {
            utterance_index: index,
            line: index + 10,
            speaker: "PAR0".to_owned(),
            timing: UtteranceTiming::Window(
                MediaWindow::new(FileMs::new(start_ms), FileMs::new(end_ms))
                    .expect("test: a non-empty utterance window"),
            ),
        }
    }

    /// The whole pipeline, end to end, with chosen vectors: one utterance that
    /// sounds like the enrolled voice, one that does not.
    #[tokio::test]
    async fn a_matching_and_a_non_matching_utterance_get_the_verdicts_the_policy_states() {
        let model = ChosenVectors {
            by_span: BTreeMap::from([
                ("enroll:INV".to_owned(), vector(vec![1.0, 0.0])),
                ("utt:0".to_owned(), vector(vec![1.0, 0.0])),
                ("utt:1".to_owned(), vector(vec![0.0, 1.0])),
            ]),
            minimum_frames: 1680,
        };
        let evidence = match identify_speakers(
            facts(),
            &enrollments(&["0-5000:INV"]),
            &[timed(0, 10_000, 12_000), timed(1, 12_000, 14_000)],
            prepared(),
            &policy(0.5),
            documented_permutation_plan(),
            &model,
        )
        .await
        {
            Ok(evidence) => evidence,
            Err(error) => panic!("the run succeeds: {error}"),
        };

        assert!(matches!(
            evidence.utterances[0].verdict,
            SpeakerVerdict::Matches { .. }
        ));
        assert!(matches!(
            evidence.utterances[1].verdict,
            SpeakerVerdict::NoMatch { .. }
        ));
        // Every utterance carries its similarity to EVERY enrolled voice, so a
        // consumer can choose its own threshold without re-running the model.
        assert_eq!(evidence.utterances[1].scores.len(), 1);
    }

    /// The same run scores the TRACKS: every speaker code becomes one voice,
    /// a refused line is counted on its track rather than imputed, and the
    /// enrolled voice's contrast names the track that is it.
    #[tokio::test]
    async fn tracks_reach_the_evidence_as_voices_with_a_contrast() {
        let mut by_span = BTreeMap::from([("enroll:INV".to_owned(), vector(vec![1.0, 0.0]))]);
        let mut utterances = Vec::new();
        for index in 0..8 {
            let (speaker, direction) = if index % 2 == 0 {
                ("PAR0", vec![1.0, 0.1 * f64::from(index)])
            } else {
                ("PAR1", vec![0.1 * f64::from(index), 1.0])
            };
            by_span.insert(format!("utt:{index}"), vector(direction));
            let start = 10_000 + 2_000 * u64::try_from(index).expect("test: a small index");
            utterances.push(TranscriptUtterance {
                speaker: speaker.to_owned(),
                ..timed(
                    usize::try_from(index).expect("test: a small index"),
                    start,
                    start + 1_500,
                )
            });
        }
        // One PAR1 line with no bullet: refused, and counted on PAR1.
        utterances.push(TranscriptUtterance {
            utterance_index: 8,
            line: 18,
            speaker: "PAR1".to_owned(),
            timing: UtteranceTiming::Untimed,
        });
        let model = ChosenVectors {
            by_span,
            minimum_frames: 1680,
        };
        let evidence = match identify_speakers(
            facts(),
            &enrollments(&["0-5000:INV"]),
            &utterances,
            prepared(),
            &policy(0.5),
            documented_permutation_plan(),
            &model,
        )
        .await
        {
            Ok(evidence) => evidence,
            Err(error) => panic!("the run succeeds: {error}"),
        };

        assert_eq!(evidence.provenance.schema_version, 2);
        assert_eq!(
            evidence.provenance.permutation,
            documented_permutation_plan()
        );
        match &evidence.tracks[..] {
            [
                TrackIdentity::Voiced {
                    track: first,
                    lines_embedded: first_embedded,
                    lines_refused: 0,
                    ..
                },
                TrackIdentity::Voiced {
                    track: second,
                    lines_embedded: second_embedded,
                    lines_refused: 1,
                    ..
                },
            ] => {
                assert_eq!(first.as_str(), "PAR0");
                assert_eq!(second.as_str(), "PAR1");
                assert_eq!(first_embedded.get(), 4);
                assert_eq!(second_embedded.get(), 4);
            }
            other => panic!("expected two voiced tracks, got {other:?}"),
        }
        match &evidence.track_contrasts[..] {
            [
                TrackContrast::Tested {
                    best,
                    runner_up,
                    p_value,
                    ..
                },
            ] => {
                assert_eq!(best.as_str(), "PAR0");
                assert_eq!(runner_up.as_str(), "PAR1");
                assert!(p_value.get() < 0.05, "p was {}", p_value.get());
            }
            other => panic!("expected one tested contrast, got {other:?}"),
        }
    }

    /// A line whose vector has no direction is one unscored line, as it was
    /// before tracks existed: it is counted as refused on its track and the
    /// run still ships its evidence.
    #[tokio::test]
    async fn a_directionless_line_is_refused_on_its_track_and_the_run_succeeds() {
        let model = ChosenVectors {
            by_span: BTreeMap::from([
                ("enroll:INV".to_owned(), vector(vec![1.0, 0.0])),
                ("utt:0".to_owned(), vector(vec![1.0, 0.0])),
                ("utt:1".to_owned(), vector(vec![0.0, 0.0])),
            ]),
            minimum_frames: 1680,
        };
        let evidence = match identify_speakers(
            facts(),
            &enrollments(&["0-5000:INV"]),
            &[timed(0, 10_000, 12_000), timed(1, 12_000, 14_000)],
            prepared(),
            &policy(0.5),
            documented_permutation_plan(),
            &model,
        )
        .await
        {
            Ok(evidence) => evidence,
            Err(error) => panic!("the run succeeds: {error}"),
        };
        assert!(matches!(
            evidence.utterances[1].verdict,
            SpeakerVerdict::Unscored { .. }
        ));
        match &evidence.tracks[..] {
            [
                TrackIdentity::Voiced {
                    lines_embedded,
                    lines_refused,
                    ..
                },
            ] => {
                assert_eq!(lines_embedded.get(), 1);
                assert_eq!(*lines_refused, 1);
            }
            other => panic!("expected one voiced track, got {other:?}"),
        }
    }

    /// Evidence combines the pinned model revision with measurements reported
    /// by the worker that actually loaded it.
    #[tokio::test]
    async fn evidence_uses_the_manifest_revision_and_worker_measurements() {
        let model = ChosenVectors {
            by_span: BTreeMap::from([
                ("enroll:INV".to_owned(), vector(vec![1.0, 0.0])),
                ("utt:0".to_owned(), vector(vec![1.0, 0.0])),
            ]),
            minimum_frames: 1680,
        };
        let evidence = match identify_speakers(
            facts(),
            &enrollments(&["0-5000:INV"]),
            &[timed(0, 10_000, 12_000)],
            prepared(),
            &policy(0.5),
            documented_permutation_plan(),
            &model,
        )
        .await
        {
            Ok(evidence) => evidence,
            Err(error) => panic!("the run succeeds: {error}"),
        };
        assert_eq!(
            evidence.provenance.embedding_model_revision.as_str(),
            pinned_embedding_revision()
                .expect("test: the packaged embedding manifest is valid")
                .as_str()
        );
        assert_eq!(evidence.provenance.embedding_dimension.get(), 2);
        assert_eq!(evidence.provenance.embedding_minimum_frames.get(), 1680);
    }

    /// An utterance inside an enrolled span is never scored: its similarity
    /// would measure the arithmetic, since the enrolled vector was computed
    /// from that very audio.
    #[tokio::test]
    async fn an_utterance_inside_the_enrollment_is_unscored_not_matched() {
        let model = ChosenVectors {
            by_span: BTreeMap::from([("enroll:INV".to_owned(), vector(vec![1.0, 0.0]))]),
            minimum_frames: 1680,
        };
        let evidence = match identify_speakers(
            facts(),
            &enrollments(&["0-5000:INV"]),
            &[timed(0, 1_000, 2_000)],
            prepared(),
            &policy(0.5),
            documented_permutation_plan(),
            &model,
        )
        .await
        {
            Ok(evidence) => evidence,
            Err(error) => panic!("the run succeeds: {error}"),
        };
        match &evidence.utterances[0].verdict {
            SpeakerVerdict::Unscored { reason } => {
                assert_eq!(reason.code(), "overlaps_enrollment");
            }
            other => panic!("expected unscored, got {other:?}"),
        }
        assert!(evidence.utterances[0].scores.is_empty());
    }

    /// An untimed utterance is reported, with its reason, rather than dropped.
    /// A file that silently omitted them would let a consumer conclude a
    /// transcript had fewer utterances than it has.
    #[tokio::test]
    async fn an_untimed_utterance_is_reported_unscored_rather_than_omitted() {
        let model = ChosenVectors {
            by_span: BTreeMap::from([("enroll:INV".to_owned(), vector(vec![1.0, 0.0]))]),
            minimum_frames: 1680,
        };
        let evidence = match identify_speakers(
            facts(),
            &enrollments(&["0-5000:INV"]),
            &[TranscriptUtterance {
                utterance_index: 0,
                line: 10,
                speaker: "PAR0".to_owned(),
                timing: UtteranceTiming::Untimed,
            }],
            prepared(),
            &policy(0.5),
            documented_permutation_plan(),
            &model,
        )
        .await
        {
            Ok(evidence) => evidence,
            Err(error) => panic!("the run succeeds: {error}"),
        };
        assert_eq!(evidence.utterances.len(), 1);
        match &evidence.utterances[0].verdict {
            SpeakerVerdict::Unscored { reason } => assert_eq!(reason.code(), "no_bullet"),
            other => panic!("expected unscored, got {other:?}"),
        }
        assert_eq!(evidence.utterances[0].start_ms, None);
    }

    /// A bullet naming audio past the end of the recording is UNSCORED with
    /// the recording's length, never embedded from whatever bytes are there.
    #[tokio::test]
    async fn a_bullet_past_the_end_of_the_recording_is_unscored_with_the_length() {
        let model = ChosenVectors {
            by_span: BTreeMap::from([("enroll:INV".to_owned(), vector(vec![1.0, 0.0]))]),
            minimum_frames: 1680,
        };
        let evidence = match identify_speakers(
            facts(),
            &enrollments(&["0-5000:INV"]),
            &[timed(0, 90_000, 92_000)],
            prepared(),
            &policy(0.5),
            documented_permutation_plan(),
            &model,
        )
        .await
        {
            Ok(evidence) => evidence,
            Err(error) => panic!("the run succeeds: {error}"),
        };
        match &evidence.utterances[0].verdict {
            SpeakerVerdict::Unscored { reason } => {
                assert_eq!(reason.code(), "audio_missing");
                assert_eq!(
                    *reason,
                    UnscoredReason::AudioMissing {
                        start_ms: 90_000,
                        end_ms: 92_000,
                        recording_ms: 60_000,
                    }
                );
            }
            other => panic!("expected unscored, got {other:?}"),
        }
    }

    /// An utterance the model refused is UNSCORED carrying the model's own
    /// minimum, never scored against a NaN vector.
    #[tokio::test]
    async fn an_utterance_the_model_refused_is_unscored_with_the_models_minimum() {
        let model = ChosenVectors {
            by_span: BTreeMap::from([
                ("enroll:INV".to_owned(), vector(vec![1.0, 0.0])),
                ("utt:0".to_owned(), SpanOutcome::TooShort { frames: 400 }),
            ]),
            minimum_frames: 1680,
        };
        let evidence = match identify_speakers(
            facts(),
            &enrollments(&["0-5000:INV"]),
            &[timed(0, 10_000, 10_020)],
            prepared(),
            &policy(0.5),
            documented_permutation_plan(),
            &model,
        )
        .await
        {
            Ok(evidence) => evidence,
            Err(error) => panic!("the run succeeds: {error}"),
        };
        match &evidence.utterances[0].verdict {
            SpeakerVerdict::Unscored { reason } => assert_eq!(
                *reason,
                UnscoredReason::TooShortForEmbedding {
                    frames: 400,
                    minimum_frames: 1680,
                }
            ),
            other => panic!("expected unscored, got {other:?}"),
        }
    }

    /// An enrollment the model cannot measure ENDS the run. Continuing would
    /// answer a different question from the one the caller asked, and every
    /// verdict in the file would be about the enrollments that happened to
    /// work.
    #[tokio::test]
    async fn an_unmeasurable_enrollment_fails_the_run_rather_than_being_dropped() {
        let model = ChosenVectors {
            by_span: BTreeMap::from([
                ("enroll:INV".to_owned(), vector(vec![1.0, 0.0])),
                (
                    "enroll:CHI".to_owned(),
                    SpanOutcome::TooShort { frames: 300 },
                ),
            ]),
            minimum_frames: 1680,
        };
        match identify_speakers(
            facts(),
            &enrollments(&["0-5000:INV", "6000-6020:CHI"]),
            &[timed(0, 10_000, 12_000)],
            prepared(),
            &policy(0.5),
            documented_permutation_plan(),
            &model,
        )
        .await
        {
            Err(SpeakerIdentityFailure::EnrollmentTooShort { label, .. }) => {
                assert_eq!(label.as_str(), "CHI");
            }
            other => panic!("expected the run to fail, got {other:?}"),
        }
    }

    /// An enrolled span outside the recording ends the run too, and names
    /// which enrollment, so an operator can fix the argument they got wrong.
    #[tokio::test]
    async fn an_enrollment_outside_the_recording_fails_the_run() {
        let model = ChosenVectors {
            by_span: BTreeMap::new(),
            minimum_frames: 1680,
        };
        match identify_speakers(
            facts(),
            &enrollments(&["90000-95000:INV"]),
            &[],
            prepared(),
            &policy(0.5),
            documented_permutation_plan(),
            &model,
        )
        .await
        {
            Err(SpeakerIdentityFailure::EnrollmentOutsideRecording { label, .. }) => {
                assert_eq!(label.as_str(), "INV");
            }
            other => panic!("expected the run to fail, got {other:?}"),
        }
    }

    /// A worker that answers about fewer spans than it was asked about fails
    /// the run rather than leaving an utterance with no verdict at all.
    #[tokio::test]
    async fn a_missing_outcome_fails_the_run() {
        let model = ChosenVectors {
            by_span: BTreeMap::from([("enroll:INV".to_owned(), vector(vec![1.0, 0.0]))]),
            minimum_frames: 1680,
        };
        match identify_speakers(
            facts(),
            &enrollments(&["0-5000:INV"]),
            &[timed(0, 10_000, 12_000)],
            prepared(),
            &policy(0.5),
            documented_permutation_plan(),
            &model,
        )
        .await
        {
            Err(SpeakerIdentityFailure::MissingOutcome { span_id }) => {
                assert_eq!(span_id, "utt:0");
            }
            other => panic!("expected the run to fail, got {other:?}"),
        }
    }

    /// Two enrolled voices that tie do not resolve to either, end to end.
    #[tokio::test]
    async fn a_tie_between_two_enrolled_voices_reaches_the_evidence_as_no_match() {
        let model = ChosenVectors {
            by_span: BTreeMap::from([
                ("enroll:INV".to_owned(), vector(vec![1.0, 0.0])),
                ("enroll:CHI".to_owned(), vector(vec![-1.0, 0.0])),
                // Orthogonal to both, so both similarities are exactly zero.
                ("utt:0".to_owned(), vector(vec![0.0, 1.0])),
            ]),
            minimum_frames: 1680,
        };
        let evidence = match identify_speakers(
            facts(),
            &enrollments(&["0-5000:INV", "6000-9000:CHI"]),
            &[timed(0, 10_000, 12_000)],
            prepared(),
            &policy(-0.5),
            documented_permutation_plan(),
            &model,
        )
        .await
        {
            Ok(evidence) => evidence,
            Err(error) => panic!("the run succeeds: {error}"),
        };
        match &evidence.utterances[0].verdict {
            SpeakerVerdict::NoMatch { best } => assert_eq!(best.labels(), ["CHI", "INV"]),
            other => panic!("a tie must not resolve to a speaker, got {other:?}"),
        }
    }
}
