//! Speaker identification by in-session enrollment.
//!
//! # What this answers, and what it deliberately does not
//!
//! Given a transcript whose utterances carry timing bullets, one or more spans
//! of the same recording known to hold a single speaker alone, and a threshold
//! the caller states, this produces one typed verdict per timed utterance: it
//! sounds like this enrolled voice, it sounds like none of them, or it could
//! not be measured and here is why.
//!
//! It does NOT rewrite the transcript. Mapping a verdict onto a CHAT speaker
//! code is a decision about a corpus's own conventions, and it is reversible
//! only if somebody kept the evidence. So the evidence IS the output, and a
//! consumer that wants to re-tier a transcript reads it and decides.
//!
//! It also does not produce accuracy. A similarity is agreement between two
//! stretches of audio under one acoustic model, and the enrollment span is a
//! human's claim about who is speaking, with its own error rate. Nothing here
//! is a gold reference and the evidence never calls a score one.
//!
//! # The type graph
//!
//! Each arrow is the only route between its two nodes, so a stage cannot be
//! skipped and no node can be built from loose parts:
//!
//! ```text
//!   CLI text  --EnrollmentSpec::parse-->  EnrollmentSpec
//!                                              |
//!                                     EnrollmentSet::new
//!                                     (non-empty, unique labels,
//!                                      non-overlapping)
//!                                              |
//!   CHAT bullet --FileMs--> MediaWindow::new --+--> a window to score
//!                                              |
//!                             PreparedPcm::locate   (the ONLY ms->frame route,
//!                                              |     and where "past the end
//!                                              v     of the audio" is detected)
//!                                          FrameSpan
//!                                              |
//!                                    worker protocol V2
//!                                              |
//!                        Embedded | TooShort  (never a NaN vector)
//!                                              |
//!                          SpeakerEmbedding::from_worker
//!                                              |
//!                          SpeakerEmbedding::similarity_to
//!                                (the ONLY producer of a score)
//!                                     |                 |
//!                     ThresholdPolicy::verdict     TrackLines::analyse
//!                  (the ONLY producer of a       (the ONLY producer of a
//!                   per-line verdict)             track voice and contrast)
//!                                     v                 v
//!                              SpeakerVerdict    TrackIdentity, TrackContrast
//! ```
//!
//! The per-line verdict and the track verdict are two readings of the same
//! vectors. The line reading answers "who said THIS"; the track reading
//! answers "which speaker code IS this enrolled voice", and carries a
//! permutation p-value instead of a margin somebody chose. See `tracks`.
//!
//! # Why the enrollment spans and the utterances share one decode
//!
//! Every span, enrollment and utterance alike, indexes into a single prepared
//! mono PCM view of the recording. Two embeddings computed from separately
//! decoded files can differ for reasons that have nothing to do with who was
//! speaking, and a similarity between them would report that difference as
//! evidence about a speaker.

pub mod embedding;
pub mod enrollment;
pub mod evidence;
pub mod frames;
pub mod model;
pub mod policy;
pub mod run;
pub mod tracks;
pub mod transcript;

pub use embedding::{
    EmbeddingDimension, IncomparableEmbeddings, MinimumEmbeddingFrames, NoCentroid, NotAnEmbedding,
    SpeakerEmbedding, ZeroEmbeddingDimension, ZeroMinimumEmbeddingFrames,
};
pub use enrollment::{
    EnrolledLabel, EnrollmentSet, EnrollmentSpec, InvalidEnrollment, InvalidEnrollmentSet,
    InvalidLabel,
};
pub use evidence::{
    EmbeddingRunFacts, RunFacts, SPEAKER_IDENTITY_SCHEMA_VERSION, SpeakerIdentityEvidence,
    SpeakerIdentityProvenance, UtteranceIdentity,
};
pub use frames::{FrameSpan, NotAPreparedDecode, OutsidePreparedAudio, PreparedPcm};
pub use model::{EmbeddingModelRevision, InvalidModelManifest, pinned_embedding_revision};
pub use policy::{
    BestScoringLabels, LabelledScore, MatchThreshold, NotASimilarity, NotAThreshold,
    SimilarityScore, SpeakerVerdict, ThresholdPolicy, UnscoredReason,
};
pub use run::{
    EmbeddingInferenceFailure, EmbeddingRequest, EmbeddingResponse, RequestedSpan, SpanOutcome,
    SpeakerEmbeddingInference, SpeakerIdentityFailure, TranscriptUtterance, UtteranceTiming,
    identify_speakers,
};
pub use tracks::{
    CentroidMargin, EmptyTrackCode, PValue, PermutationCount, PermutationPlan, TrackAnalysis,
    TrackAnalysisFailure, TrackCode, TrackContrast, TrackIdentity, TrackLines, ZeroPermutations,
    documented_permutation_plan,
};
pub use transcript::{TierSelection, read_utterances};
