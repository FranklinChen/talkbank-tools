//! Preserve acoustic choices while retaining their source-bound justification.

use super::*;
use batchalign_transform::asr_postprocess::AsrWord;

struct IndexedSegment<'a> {
    source: &'a SpeakerSegment,
    speaker: DiarizationSpeakerIndex,
}

/// Empty evidence cannot supply a dedicated-label coordinate or a fallback.
struct NonEmptyDiarization<'a> {
    first: IndexedSegment<'a>,
    rest: Vec<IndexedSegment<'a>>,
    label_count: usize,
}

impl<'a> NonEmptyDiarization<'a> {
    fn admit(segments: &'a [SpeakerSegment]) -> Option<Self> {
        let (first, rest) = segments.split_first()?;
        let coordinates = DiarizationLabelCoordinates::from_labels(
            segments.iter().map(|segment| segment.speaker.as_str()),
        );
        let index = |source: &'a SpeakerSegment| {
            // Both the coordinates and segments belong to this producer input.
            #[allow(clippy::expect_used)]
            let speaker = coordinates
                .index_for(&source.speaker)
                .expect("source label belongs to the closed diarization coordinates");
            IndexedSegment { source, speaker }
        };
        Some(Self {
            first: index(first),
            rest: rest.iter().map(index).collect(),
            label_count: coordinates.len(),
        })
    }

    fn segments(&self) -> impl Iterator<Item = &IndexedSegment<'a>> {
        std::iter::once(&self.first).chain(self.rest.iter())
    }
}

struct ChosenAssignment {
    speaker: DiarizationSpeakerIndex,
    basis: AssignmentBasis,
}

struct WordProjection {
    chosen: Option<ChosenAssignment>,
    timing: TimingSupport,
}

#[derive(Clone, Copy)]
struct AssignmentWitness {
    source: SourceWordAddress,
    speaker: DiarizationSpeakerIndex,
}

/// Greatest accumulated overlap; lexical-label ties; nearest-segment gaps.
/// Untimed tokens retain the existing backward/forward attachment policy.
/// Extraction requires explicit best-effort admission, not an accuracy claim.
pub fn project_speakers_onto_chunks(
    chunks: Vec<PreparedMonologueChunk>,
    segments: &[SpeakerSegment],
) -> SpeakerProjection {
    let mut evidence = SpeakerProjectionEvidence::source_bound(&chunks, segments);
    let Some(diarization) = NonEmptyDiarization::admit(segments) else {
        for (chunk_index, chunk) in chunks.iter().enumerate() {
            for word_index in 0..chunk.words.len() {
                evidence.record(WordAssignmentEvidence {
                    source: SourceWordAddress {
                        chunk: chunk_index,
                        word: word_index,
                    },
                    coordinate: SpeakerCoordinate::Asr(chunk.speaker.0),
                    timing: TimingSupport::NotObserved,
                    basis: AssignmentBasis::RetainedAsr,
                });
            }
        }
        return SpeakerProjection { chunks, evidence };
    };
    let mut projected_chunks = Vec::new();
    for (chunk_index, chunk) in chunks.into_iter().enumerate() {
        let mut assignments: Vec<WordProjection> = chunk
            .words
            .iter()
            .map(|word| project_timed_word(word, &diarization))
            .collect();
        let mut preceding: Option<AssignmentWitness> = None;
        for (word_index, assignment) in assignments.iter_mut().enumerate() {
            if let Some(chosen) = &assignment.chosen {
                preceding = Some(AssignmentWitness {
                    source: SourceWordAddress {
                        chunk: chunk_index,
                        word: word_index,
                    },
                    speaker: chosen.speaker,
                });
            } else if let Some(witness) = preceding {
                assignment.chosen = Some(ChosenAssignment {
                    speaker: witness.speaker,
                    basis: AssignmentBasis::PreviousWord {
                        witness: witness.source,
                    },
                });
            }
        }
        let mut following: Option<AssignmentWitness> = None;
        for (word_index, assignment) in assignments.iter_mut().enumerate().rev() {
            if let Some(chosen) = &assignment.chosen {
                following = Some(AssignmentWitness {
                    source: SourceWordAddress {
                        chunk: chunk_index,
                        word: word_index,
                    },
                    speaker: chosen.speaker,
                });
            } else if let Some(witness) = following {
                assignment.chosen = Some(ChosenAssignment {
                    speaker: witness.speaker,
                    basis: AssignmentBasis::FollowingWord {
                        witness: witness.source,
                    },
                });
            }
        }
        let mut current_speaker: Option<DiarizationSpeakerIndex> = None;
        let mut current_words = Vec::new();
        for (word_index, (word, assignment)) in chunk.words.into_iter().zip(assignments).enumerate()
        {
            // Nonempty evidence establishes coordinate zero, but not support
            // for this word. Record the default instead of certifying it.
            let chosen = assignment.chosen.unwrap_or(ChosenAssignment {
                speaker: DiarizationSpeakerIndex(0),
                basis: AssignmentBasis::FirstLabelDefault,
            });
            evidence.record(WordAssignmentEvidence {
                source: SourceWordAddress {
                    chunk: chunk_index,
                    word: word_index,
                },
                coordinate: SpeakerCoordinate::Diarization(chosen.speaker.0),
                timing: assignment.timing,
                basis: chosen.basis,
            });
            if let Some(current) = current_speaker.filter(|current| *current != chosen.speaker) {
                projected_chunks.push(PreparedMonologueChunk {
                    speaker: current.flatten(),
                    words: std::mem::take(&mut current_words),
                });
                evidence.record_boundary();
            }
            current_speaker = Some(chosen.speaker);
            current_words.push(word);
        }
        if let Some(speaker) = current_speaker {
            projected_chunks.push(PreparedMonologueChunk {
                speaker: speaker.flatten(),
                words: current_words,
            });
        }
    }
    SpeakerProjection {
        chunks: projected_chunks,
        evidence,
    }
}

fn project_timed_word(word: &AsrWord, diarization: &NonEmptyDiarization<'_>) -> WordProjection {
    if word.start_ms.is_none() && word.end_ms.is_none() {
        return WordProjection {
            chosen: None,
            timing: TimingSupport::Untimed,
        };
    }
    let (Some(start), Some(end)) = (word.start_ms, word.end_ms) else {
        return WordProjection {
            chosen: None,
            timing: TimingSupport::InvalidInterval,
        };
    };
    if start < 0 || end <= start {
        return WordProjection {
            chosen: None,
            timing: TimingSupport::InvalidInterval,
        };
    }
    let (start, end) = (start as u64, end as u64);
    let mut overlaps = vec![0u64; diarization.label_count];
    for segment in diarization.segments() {
        let overlap_start = start.max(segment.source.interval.start_millis());
        let overlap_end = end.min(segment.source.interval.end_millis());
        if overlap_end > overlap_start {
            overlaps[segment.speaker.0] += overlap_end - overlap_start;
        }
    }
    let speakers = overlaps.iter().filter(|overlap| **overlap > 0).count();
    if speakers == 0 {
        let distance = |segment: &IndexedSegment<'_>| {
            let segment_start = segment.source.interval.start_millis();
            if end <= segment_start {
                segment_start - end
            } else {
                start.saturating_sub(segment.source.interval.end_millis())
            }
        };
        let mut nearest = &diarization.first;
        let mut segment_index = 0;
        let mut distance_ms = distance(nearest);
        for (index, segment) in diarization.rest.iter().enumerate() {
            let candidate = distance(segment);
            if candidate < distance_ms {
                nearest = segment;
                segment_index = index + 1;
                distance_ms = candidate;
            }
        }
        return WordProjection {
            chosen: Some(ChosenAssignment {
                speaker: nearest.speaker,
                basis: AssignmentBasis::NearestSegment,
            }),
            timing: TimingSupport::Gap {
                segment: segment_index,
                distance_ms,
            },
        };
    }
    // The admitted nonempty source establishes at least one label. Strictly
    // greater replacement preserves the lexical-label tie-break.
    let mut winner = 0;
    for index in 1..overlaps.len() {
        if overlaps[index] > overlaps[winner] {
            winner = index;
        }
    }
    let winning_overlap_ms = overlaps[winner];
    let tied_winners = overlaps
        .iter()
        .filter(|overlap| **overlap == winning_overlap_ms)
        .count();
    WordProjection {
        chosen: Some(ChosenAssignment {
            speaker: DiarizationSpeakerIndex(winner),
            basis: AssignmentBasis::DirectOverlap,
        }),
        timing: TimingSupport::Overlap {
            speakers,
            tied_winners,
            winning_overlap_ms,
        },
    }
}
