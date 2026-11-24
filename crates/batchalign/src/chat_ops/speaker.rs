//! Speaker diarization projection onto timed ASR words.
//!
//! Dedicated diarization is available before utterance segmentation. This
//! module projects its segments onto normalized ASR words and splits prepared
//! chunks at observed speaker boundaries, so later retokenization cannot join
//! words attributed to different speakers into one CHAT utterance.

use std::collections::BTreeMap;

use batchalign_transform::asr_postprocess::{PreparedMonologueChunk, SpeakerIndex};

mod evidence;
mod projection;
pub use evidence::{
    AdmittedSpeakerProjectionEvidence, AssignmentBasis, SourceWordAddress,
    SpeakerAssignmentSummary, SpeakerCoordinate, SpeakerProjectionEvidence,
    SpeakerProjectionPolicy, TimingSupport, WordAssignmentEvidence,
};
pub use projection::project_speakers_onto_chunks;

/// One raw diarization segment to project onto timed ASR words: its span
/// as the admitted interval it was read as (ordered and in range), never
/// lowered back to a pair of bare numbers.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct SpeakerSegment {
    /// The segment's span of the media.
    pub interval: batchalign_types::interval::AdmittedInterval,
    /// Stable speaker label emitted by the model host.
    pub speaker: String,
}

/// Counts that make imperfect diarization projection observable.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SpeakerProjectionStats {
    /// Timed words overlapped by more than one diarization speaker.
    pub contested_timed_words: usize,
    /// Timed words with no overlapping diarization segment.
    pub unattested_timed_words: usize,
    /// New chunk boundaries introduced by a projected speaker change.
    pub speaker_boundaries: usize,
}

/// Prepared chunks after diarization has constrained speaker boundaries.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeakerProjection {
    chunks: Vec<PreparedMonologueChunk>,
    evidence: SpeakerProjectionEvidence,
}

/// Chunks may leave observation only after an explicit assignment policy.
#[derive(Debug, Clone, PartialEq)]
pub struct AdmittedSpeakerProjection {
    chunks: Vec<PreparedMonologueChunk>,
    evidence: AdmittedSpeakerProjectionEvidence,
}

pub(crate) struct SpeakerProjectionParts {
    pub(crate) chunks: Vec<PreparedMonologueChunk>,
    pub(crate) evidence: AdmittedSpeakerProjectionEvidence,
}

impl SpeakerProjection {
    /// Inspect observations without granting access to projected chunks.
    #[must_use]
    pub const fn observations(&self) -> &SpeakerProjectionEvidence {
        &self.evidence
    }
    /// Derive the historical projection counters from the observations.
    #[must_use]
    pub fn stats(&self) -> SpeakerProjectionStats {
        self.evidence.stats()
    }

    /// Consume observations under an explicit best-effort output policy.
    #[must_use]
    pub fn admit(self, policy: SpeakerProjectionPolicy) -> AdmittedSpeakerProjection {
        AdmittedSpeakerProjection {
            chunks: self.chunks,
            evidence: self.evidence.admit(policy),
        }
    }
}

impl AdmittedSpeakerProjection {
    /// Read chunks admitted with the accompanying review evidence.
    #[must_use]
    pub fn chunks(&self) -> &[PreparedMonologueChunk] {
        &self.chunks
    }
    /// Read the producer-owned evidence and its selected policy.
    #[must_use]
    pub const fn evidence(&self) -> &AdmittedSpeakerProjectionEvidence {
        &self.evidence
    }

    pub(crate) fn into_parts(self) -> SpeakerProjectionParts {
        SpeakerProjectionParts {
            chunks: self.chunks,
            evidence: self.evidence,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DiarizationSpeakerIndex(usize);

impl DiarizationSpeakerIndex {
    const fn flatten(self) -> SpeakerIndex {
        SpeakerIndex(self.0)
    }

    /// Stable anonymous-track number used by speaker-turn artifacts.
    pub(crate) const fn as_usize(self) -> usize {
        self.0
    }
}

/// Deterministic coordinates for model-native diarization labels.
///
/// Both word projection and retained turn artifacts must use this same map.
/// Otherwise `PAR0` in generated CHAT can refer to a different acoustic voice
/// from `PAR0` in the evidence file. Lexical ordering is stable even when a
/// provider returns the same turns in a different sequence.
pub(crate) struct DiarizationLabelCoordinates {
    index_by_label: BTreeMap<String, DiarizationSpeakerIndex>,
}

impl DiarizationLabelCoordinates {
    /// Build a closed coordinate system from every observed model label.
    pub(crate) fn from_labels<'a>(labels: impl IntoIterator<Item = &'a str>) -> Self {
        let lexical_labels = labels.into_iter().map(|label| (label, ())).collect();
        let index_by_label = Self::number(lexical_labels)
            .map(|(index, (label, ()))| (label.to_owned(), index))
            .collect();
        Self { index_by_label }
    }

    /// Number worker intervals by their real label in lexical order, carrying
    /// the typed interval into the resulting track without another map lookup.
    pub(crate) fn numbered_turns(
        segments: &[crate::types::worker_v2::SpeakerSegmentV2],
    ) -> impl Iterator<
        Item = (
            DiarizationSpeakerIndex,
            batchalign_types::interval::AdmittedInterval,
        ),
    > {
        let mut by_label = BTreeMap::<&str, Vec<_>>::new();
        for segment in segments {
            by_label
                .entry(segment.speaker.as_str())
                .or_default()
                .push(segment.interval);
        }
        Self::number(by_label).flat_map(|(index, (_, intervals))| {
            intervals.into_iter().map(move |interval| (index, interval))
        })
    }

    fn number<T>(
        ordered: BTreeMap<&str, T>,
    ) -> impl Iterator<Item = (DiarizationSpeakerIndex, (&str, T))> {
        ordered
            .into_iter()
            .enumerate()
            .map(|(index, source)| (DiarizationSpeakerIndex(index), source))
    }

    /// Resolve one label into the shared anonymous-speaker coordinate system.
    pub(crate) fn index_for(&self, label: &str) -> Option<DiarizationSpeakerIndex> {
        self.index_by_label.get(label).copied()
    }

    fn len(&self) -> usize {
        self.index_by_label.len()
    }
}

#[cfg(test)]
mod tests;
