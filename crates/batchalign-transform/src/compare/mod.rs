//! Transcript comparison: main vs gold-standard reference.
//!
//! Extracts words from both transcripts, normalizes them via
//! [`wer_conform::conform_words`], runs DP alignment, and produces
//! per-utterance comparison annotations with accuracy metrics.
//!
//! This module keeps compare concerns split by responsibility:
//! - [`model`] defines the workflow data structures
//! - [`engine`] owns alignment and bundle construction
//! - [`metrics`] owns aggregate counters and CSV materialization
//! - [`serialize`] owns `%xsrep` / `%xsmor` rendering
//! - [`materialize`] owns CHAT-tier injection and gold projection
//!
//! This is the Rust implementation of Python's `CompareEngine` +
//! `CompareAnalysisEngine` from batchalign2.

mod artifact;
mod cross_mode;
mod cross_run;
mod engine;
mod materialize;
mod metrics;
mod model;
mod plan;
mod pos;
mod serialize;

pub use self::artifact::{
    AggregatePolicy, ArtifactContractError, ArtifactDigest, ArtifactPair, ComparisonPlan,
    ComparisonSubject, HumanIdentity, MachineIdentity, PairingPolicy, ProducedRun,
    RUN_MANIFEST_SCHEMA_VERSION, RelativeArtifactPath, RunArtifactRoot, RunIdentity, RunManifest,
    ValidatedAlignmentPlan, ValidatedArtifactPair, ValidatedComparisonPlan, ValidatedMorphotagPlan,
    ValidatedProducedRun, ValidatedTranscriptionPlan,
};
pub use self::cross_mode::{
    AlignmentPairResult, AlignmentTokenDifference, LexicalMismatchCount, MorphotagDifference,
    MorphotagPairResult, MorphotagTokenDifference, PairFailureReason, PairOutcome,
    TimingDistribution, TokenTiming, TokenTimingState, WorSlotDrift,
    compare_validated_alignment_plan, compare_validated_morphotag_plan,
    compare_validated_transcription_pairs,
};
pub use self::cross_run::{
    AgreementMetrics, AmbiguousSpeakerMaps, ArtifactTranscriptionComparison,
    CROSS_RUN_REPORT_SCHEMA_VERSION, CrossRunReportError, CrossRunTranscriptionPlanReport,
    CrossRunTranscriptionReport, CrossRunTranscriptionReportArtifact, SpeakerAgreement,
    SpeakerCorrespondence, SpeakerMap, SpeakerMapApplicationError, SpeakerMapBuildError,
    SpeakerMapCandidate, compare_transcripts_by_speaker, compare_transcripts_with_exclusions,
    compare_transcripts_with_speaker_map, compare_validated_transcription_plan,
    serialize_cross_run_csv, serialize_cross_run_json,
};
pub use self::engine::compare;
pub use self::materialize::{clear_comparison, inject_comparison, project_gold_structurally};
pub use self::metrics::{
    CompareCsvHeader, CompareMetricName, CompareMetricValue, CompareMetricsCsvRow,
    CompareMetricsCsvTable, ComparePosMetricKind, format_metrics_csv,
};
pub use self::model::{
    CompareMetrics, CompareResult, CompareStatus, CompareToken, ComparisonBundle, GoldCoverage,
    GoldWordMatch, PosErrorCounts, UtteranceComparison,
};
pub use self::plan::{ComparisonPlanDocument, ResolvedComparisonPlan};
pub use self::serialize::{
    ComparePosLabel, CompareSerializationError, CompareSurfaceToken, CompareTierItem,
    CompareTierLabel, CompareTierMarker, CompareUserDefinedTier, XsmorTierContent,
    XsrepTierContent, format_xsmor, format_xsrep,
};

#[cfg(test)]
mod tests;
