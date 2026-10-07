//! Source-bound word assignment observations and explicit best-effort admission.

use batchalign_transform::asr_postprocess::{PreparedMonologueChunk, WordKind};
use serde::Serialize;

use super::{SpeakerProjectionStats, SpeakerSegment};

/// Address in the prepared ASR input, before speaker or utterance splitting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct SourceWordAddress {
    pub(super) chunk: usize,
    pub(super) word: usize,
}

impl SourceWordAddress {
    /// Zero-based chunk address before speaker and utterance splitting.
    pub fn chunk(&self) -> usize {
        self.chunk
    }
    /// Zero-based token-slot address within that prepared chunk.
    pub fn word(&self) -> usize {
        self.word
    }
}

/// The coordinate space matters: ASR index zero is not diarization index zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(tag = "space", content = "index", rename_all = "snake_case")]
pub enum SpeakerCoordinate {
    /// The original ASR producer's index; no dedicated assignment is claimed.
    Asr(usize),
    /// Lexically ordered model label in the closed dedicated evidence space.
    Diarization(usize),
}

/// Observed timing support, distinct from the policy that selected a label.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TimingSupport {
    /// No dedicated intervals were supplied to the projection producer.
    NotObserved,
    /// Neither timing endpoint was available for this token slot.
    Untimed,
    /// Timing was partial, negative or not strictly increasing.
    InvalidInterval,
    /// No positive overlap; a nearest segment supplied a best-effort label.
    Gap {
        /// Zero-based position in the exact ordered segment input.
        segment: usize,
        /// Gap between the token interval and selected segment in milliseconds.
        distance_ms: u64,
    },
    /// Positive accumulated overlap was observed in dedicated label space.
    Overlap {
        /// Number of labels with positive overlap, including losing labels.
        speakers: usize,
        /// Number of labels tied for the greatest accumulated overlap.
        tied_winners: usize,
        /// Sum of the winning label's overlapping segment milliseconds.
        winning_overlap_ms: u64,
    },
}

/// The producer's actual selection or fallback, not an acoustic accuracy claim.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AssignmentBasis {
    /// Greatest accumulated overlap, with lexical-label tie-breaking.
    DirectOverlap,
    /// Closest segment interval, with original segment-order tie-breaking.
    NearestSegment,
    /// Attach to a previously resolved token slot in this prepared chunk.
    PreviousWord {
        /// The source address from which the label was inherited.
        witness: SourceWordAddress,
    },
    /// Attach to a following resolved token slot when no prior label exists.
    FollowingWord {
        /// The source address from which the label was inherited.
        witness: SourceWordAddress,
    },
    /// A wholly unresolved chunk takes the first lexical dedicated label.
    FirstLabelDefault,
    /// Empty dedicated evidence preserves the original ASR speaker index.
    RetainedAsr,
}

/// One prepared input token's coordinates, timing support and assignment basis.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct WordAssignmentEvidence {
    pub(super) source: SourceWordAddress,
    pub(super) coordinate: SpeakerCoordinate,
    pub(super) timing: TimingSupport,
    pub(super) basis: AssignmentBasis,
}

impl WordAssignmentEvidence {
    /// Locate this token slot in the exact prepared input.
    pub fn source(&self) -> SourceWordAddress {
        self.source
    }
    /// Read the selected index together with its coordinate space.
    pub fn coordinate(&self) -> SpeakerCoordinate {
        self.coordinate
    }
    /// Inspect the available timing evidence independently of fallback policy.
    pub fn timing(&self) -> &TimingSupport {
        &self.timing
    }
    /// Inspect the producer's actual assignment decision.
    pub fn basis(&self) -> &AssignmentBasis {
        &self.basis
    }
}

/// Observations are constructed by projection, never by its consumers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SpeakerProjectionEvidence {
    input_chunks_blake3: String,
    segments_blake3: String,
    assignments: Vec<WordAssignmentEvidence>,
    speaker_boundaries: usize,
}

/// Disjoint counts of prepared token slots, including untimed punctuation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct SpeakerAssignmentSummary {
    /// Total original prepared token slots; not a lexical accuracy denominator.
    pub words: usize,
    /// Slots with positive overlap for exactly one dedicated label.
    pub directly_supported: usize,
    /// Slots with positive overlap for more than one dedicated label.
    pub contested: usize,
    /// Nearest-segment and neighboring-token fallback assignments.
    pub inferred: usize,
    /// Wholly unresolved slots assigned the first lexical dedicated label.
    pub defaulted: usize,
    /// Slots preserving ASR indices because dedicated evidence was empty.
    pub retained_asr: usize,
}

/// This policy admits existing choices only WITH retained review evidence.
/// It does not certify acoustic accuracy or a named person's identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SpeakerProjectionPolicy {
    /// Preserve existing acoustic choices and retain any required review caveat.
    BestEffortWithReviewEvidenceV1,
}

/// Producer-created observations consumed under an explicit output policy.
/// Serialization derives its summary; deserialization cannot forge admission.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdmittedSpeakerProjectionEvidence {
    policy: SpeakerProjectionPolicy,
    evidence: SpeakerProjectionEvidence,
}

impl SpeakerProjectionEvidence {
    pub(super) fn source_bound(
        chunks: &[PreparedMonologueChunk],
        segments: &[SpeakerSegment],
    ) -> Self {
        let mut input = blake3::Hasher::new();
        input.update(b"speaker-word-input-v1\0");
        input.update(&(chunks.len() as u64).to_le_bytes());
        for chunk in chunks {
            input.update(&(chunk.speaker.0 as u64).to_le_bytes());
            input.update(&(chunk.words.len() as u64).to_le_bytes());
            for word in &chunk.words {
                let text = word.text.as_str().as_bytes();
                input.update(&(text.len() as u64).to_le_bytes());
                input.update(text);
                for position in [word.start_ms, word.end_ms] {
                    match position {
                        Some(value) => {
                            input.update(&[1]);
                            input.update(&value.to_le_bytes());
                        }
                        None => {
                            input.update(&[0]);
                        }
                    }
                }
                input.update(&[match word.kind {
                    WordKind::Regular => 0,
                    WordKind::Retrace => 1,
                }]);
            }
        }
        let mut acoustic = blake3::Hasher::new();
        acoustic.update(b"speaker-word-segments-v1\0");
        acoustic.update(&(segments.len() as u64).to_le_bytes());
        for segment in segments {
            acoustic.update(&segment.interval.start_millis().to_le_bytes());
            acoustic.update(&segment.interval.end_millis().to_le_bytes());
            acoustic.update(&(segment.speaker.len() as u64).to_le_bytes());
            acoustic.update(segment.speaker.as_bytes());
        }
        Self {
            input_chunks_blake3: input.finalize().to_hex().to_string(),
            segments_blake3: acoustic.finalize().to_hex().to_string(),
            assignments: Vec::new(),
            speaker_boundaries: 0,
        }
    }

    pub(super) fn record(&mut self, assignment: WordAssignmentEvidence) {
        self.assignments.push(assignment);
    }

    pub(super) fn record_boundary(&mut self) {
        self.speaker_boundaries += 1;
    }

    /// Read source-ordered observations, one per prepared input token slot.
    pub fn assignments(&self) -> &[WordAssignmentEvidence] {
        &self.assignments
    }

    /// Derive the historical aggregate projection counters.
    pub fn stats(&self) -> SpeakerProjectionStats {
        SpeakerProjectionStats {
            contested_timed_words: self.assignments.iter().filter(|entry| {
                matches!(entry.timing, TimingSupport::Overlap { speakers, .. } if speakers > 1)
            }).count(),
            unattested_timed_words: self.assignments.iter().filter(|entry| {
                matches!(entry.timing, TimingSupport::InvalidInterval | TimingSupport::Gap { .. })
            }).count(),
            speaker_boundaries: self.speaker_boundaries,
        }
    }

    /// Derive disjoint support/fallback counts rather than cache caller claims.
    pub fn summary(&self) -> SpeakerAssignmentSummary {
        let mut summary = SpeakerAssignmentSummary {
            words: self.assignments.len(),
            directly_supported: 0,
            contested: 0,
            inferred: 0,
            defaulted: 0,
            retained_asr: 0,
        };
        for entry in &self.assignments {
            match entry.basis {
                AssignmentBasis::DirectOverlap => match entry.timing {
                    TimingSupport::Overlap { speakers: 1, .. } => summary.directly_supported += 1,
                    _ => summary.contested += 1,
                },
                AssignmentBasis::FirstLabelDefault => summary.defaulted += 1,
                AssignmentBasis::RetainedAsr => summary.retained_asr += 1,
                _ => summary.inferred += 1,
            }
        }
        summary
    }

    /// Whether any slot lacks positive overlap with exactly one label.
    /// False does not certify model accuracy or named-person identity.
    pub fn needs_review(&self) -> bool {
        self.summary().directly_supported != self.assignments.len()
    }

    pub(super) fn admit(
        self,
        policy: SpeakerProjectionPolicy,
    ) -> AdmittedSpeakerProjectionEvidence {
        AdmittedSpeakerProjectionEvidence {
            policy,
            evidence: self,
        }
    }
}

impl AdmittedSpeakerProjectionEvidence {
    /// Read the exact observations admitted by the selected output policy.
    pub fn observations(&self) -> &SpeakerProjectionEvidence {
        &self.evidence
    }
    /// Derive the review requirement from the producer-owned observations.
    pub fn needs_review(&self) -> bool {
        self.evidence.needs_review()
    }

    /// The warning is derived from the admitted observations, not caller prose.
    pub(crate) fn review_warning(&self) -> Option<String> {
        if !self.needs_review() {
            return None;
        }
        let summary = self.evidence.summary();
        Some(format!(
            "WARNING: Speaker projection requires review (best-effort-v1): contested={}, inferred={}, defaulted={}, retained-ASR={}. These assignments are not complete acoustic evidence.",
            summary.contested, summary.inferred, summary.defaulted, summary.retained_asr
        ))
    }
}

impl Serialize for AdmittedSpeakerProjectionEvidence {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        #[derive(Serialize)]
        struct View<'a> {
            schema_version: u32,
            policy: SpeakerProjectionPolicy,
            needs_review: bool,
            summary: SpeakerAssignmentSummary,
            evidence: &'a SpeakerProjectionEvidence,
        }
        View {
            schema_version: 1,
            policy: self.policy,
            needs_review: self.needs_review(),
            summary: self.evidence.summary(),
            evidence: &self.evidence,
        }
        .serialize(serializer)
    }
}
