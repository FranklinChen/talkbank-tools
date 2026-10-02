//! Versioned causal receipts for speaker-evidence projections.

use serde::Serialize;

use super::{
    SpeakerBackendV2, SpeakerEvidenceMissReason, SpeakerEvidenceResolution, SpeakerEvidenceSource,
    SpeakerSegmentV2,
};

pub(super) const SPEAKER_SEGMENT_DIGEST_REVISION: &str = "speaker-segments-blake3-v1";

/// Stable revision of the local raw-evidence-to-segment projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub(crate) enum SpeakerProjectionRevision {
    #[serde(rename = "speaker-evidence-to-segments-v1")]
    SegmentsV1,
}

/// Causal result of the cache-or-infer decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum SpeakerEvidenceCacheOutcome {
    ReplayedDerived,
    DerivedFromRaw,
    InferredNotFound,
    InferredForcedRefresh,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(super) struct SpeakerEvidenceTraceSeed {
    pub(super) trace_schema_version: u32,
    pub(super) source_media_blake3: String,
    pub(super) audio_preparation_revision: &'static str,
    pub(super) backend: SpeakerBackendV2,
    pub(super) expected_speakers: Option<u32>,
    pub(super) model_revision: String,
    pub(super) raw_evidence_key: String,
    pub(super) normalization_revision: String,
    pub(super) derived_evidence_key: String,
}

/// Content identity of the exact normalized segments used downstream.
///
/// The digest is constructed only from validated evidence, after segment
/// geometry has passed validation. It is independent of JSON formatting and
/// cache location.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub(super) struct SpeakerSegmentsDigest(String);

impl SpeakerSegmentsDigest {
    pub(super) fn from_segments(segments: &[SpeakerSegmentV2]) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(SPEAKER_SEGMENT_DIGEST_REVISION.as_bytes());
        hasher.update(&[0]);
        hasher.update(&(segments.len() as u64).to_le_bytes());
        for segment in segments {
            hasher.update(&segment.interval.start_millis().to_le_bytes());
            hasher.update(&segment.interval.end_millis().to_le_bytes());
            let speaker = segment.speaker.as_bytes();
            hasher.update(&(speaker.len() as u64).to_le_bytes());
            hasher.update(speaker);
        }
        Self(hasher.finalize().to_hex().to_string())
    }
}

/// Versioned causal receipt for one dedicated-speaker projection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct SpeakerEvidenceTrace {
    #[serde(flatten)]
    request: SpeakerEvidenceTraceSeed,
    cache_outcome: SpeakerEvidenceCacheOutcome,
    projection_revision: SpeakerProjectionRevision,
    segment_digest_revision: &'static str,
    projected_segment_count: usize,
    projected_segments_blake3: SpeakerSegmentsDigest,
}

#[cfg(test)]
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct SpeakerEvidenceSemanticProjection<'a> {
    request: &'a SpeakerEvidenceTraceSeed,
    projection_revision: SpeakerProjectionRevision,
    projected_segment_count: usize,
    projected_segments_blake3: &'a SpeakerSegmentsDigest,
}

impl SpeakerEvidenceTrace {
    #[cfg(test)]
    pub(crate) fn cache_outcome(&self) -> SpeakerEvidenceCacheOutcome {
        self.cache_outcome
    }

    #[cfg(test)]
    pub(crate) fn semantic_projection(&self) -> SpeakerEvidenceSemanticProjection<'_> {
        SpeakerEvidenceSemanticProjection {
            request: &self.request,
            projection_revision: self.projection_revision,
            projected_segment_count: self.projected_segment_count,
            projected_segments_blake3: &self.projected_segments_blake3,
        }
    }
}

impl SpeakerEvidenceResolution {
    pub(crate) fn trace(
        &self,
        projection_revision: SpeakerProjectionRevision,
    ) -> SpeakerEvidenceTrace {
        let cache_outcome = match self.source {
            SpeakerEvidenceSource::ReplayedDerived => SpeakerEvidenceCacheOutcome::ReplayedDerived,
            SpeakerEvidenceSource::DerivedFromRaw => SpeakerEvidenceCacheOutcome::DerivedFromRaw,
            SpeakerEvidenceSource::Inferred(SpeakerEvidenceMissReason::NotFound) => {
                SpeakerEvidenceCacheOutcome::InferredNotFound
            }
            SpeakerEvidenceSource::Inferred(SpeakerEvidenceMissReason::ForcedRefresh) => {
                SpeakerEvidenceCacheOutcome::InferredForcedRefresh
            }
        };
        SpeakerEvidenceTrace {
            request: self.trace_seed.clone(),
            cache_outcome,
            projection_revision,
            segment_digest_revision: SPEAKER_SEGMENT_DIGEST_REVISION,
            projected_segment_count: self.evidence.segments.len(),
            projected_segments_blake3: self.evidence.segments_digest.clone(),
        }
    }
}
