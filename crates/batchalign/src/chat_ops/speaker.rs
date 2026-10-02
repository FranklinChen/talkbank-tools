//! Speaker diarization projection onto timed ASR words.
//!
//! Dedicated diarization is available before utterance segmentation. This
//! module projects its segments onto normalized ASR words and splits prepared
//! chunks at observed speaker boundaries, so later retokenization cannot join
//! words attributed to different speakers into one CHAT utterance.

use std::collections::BTreeMap;

use batchalign_transform::asr_postprocess::{AsrWord, PreparedMonologueChunk, SpeakerIndex};

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
    /// Chunks carrying a single resolved speaker coordinate each.
    pub chunks: Vec<PreparedMonologueChunk>,
    /// Projection diagnostics.
    pub stats: SpeakerProjectionStats,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DiarizationSpeakerIndex(usize);

impl DiarizationSpeakerIndex {
    fn flatten(self) -> SpeakerIndex {
        SpeakerIndex(self.0)
    }

    /// Stable anonymous-track number used by speaker-turn artifacts.
    pub(crate) fn as_usize(self) -> usize {
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
        let index_by_label = labels
            .into_iter()
            .map(str::to_owned)
            .collect::<std::collections::BTreeSet<String>>()
            .into_iter()
            .enumerate()
            .map(|(index, label)| (label, DiarizationSpeakerIndex(index)))
            .collect();
        Self { index_by_label }
    }

    /// Resolve one label into the shared anonymous-speaker coordinate system.
    pub(crate) fn index_for(&self, label: &str) -> Option<DiarizationSpeakerIndex> {
        self.index_by_label.get(label).copied()
    }

    fn len(&self) -> usize {
        self.index_by_label.len()
    }
}

/// Project diarization evidence onto words before utterance segmentation.
///
/// A timed word is assigned to the speaker with the greatest total overlap.
/// Ties resolve by lexical model-label order, the same deterministic
/// coordinate system used by the retained speaker-turn artifact.
/// Untimed tokens inherit the preceding resolved speaker, then the following
/// one when no preceding timed token exists. A timed word in a diarization gap
/// takes the nearest dedicated segment's label. Once nonempty dedicated
/// evidence exists, output cannot re-enter the unrelated ASR label space.
pub fn project_speakers_onto_chunks(
    chunks: Vec<PreparedMonologueChunk>,
    segments: &[SpeakerSegment],
) -> SpeakerProjection {
    if segments.is_empty() {
        return SpeakerProjection {
            chunks,
            stats: SpeakerProjectionStats::default(),
        };
    }

    let label_coordinates = DiarizationLabelCoordinates::from_labels(
        segments.iter().map(|segment| segment.speaker.as_str()),
    );
    let mut stats = SpeakerProjectionStats::default();
    let mut projected_chunks = Vec::new();

    for chunk in chunks {
        let default_diarized = DiarizationSpeakerIndex(0);
        let mut assignments: Vec<Option<DiarizationSpeakerIndex>> = chunk
            .words
            .iter()
            .map(|word| {
                if word.start_ms.is_none() && word.end_ms.is_none() {
                    None
                } else {
                    project_timed_word(word, segments, &label_coordinates, &mut stats)
                }
            })
            .collect();

        let mut preceding = None;
        for assignment in &mut assignments {
            if assignment.is_some() {
                preceding = *assignment;
            } else if preceding.is_some() {
                *assignment = preceding;
            }
        }
        let mut following = None;
        for assignment in assignments.iter_mut().rev() {
            if assignment.is_some() {
                following = *assignment;
            } else if following.is_some() {
                *assignment = following;
            }
        }

        let mut current_speaker: Option<DiarizationSpeakerIndex> = None;
        let mut current_words: Vec<AsrWord> = Vec::new();
        for (word, assignment) in chunk.words.into_iter().zip(assignments) {
            let speaker = assignment.unwrap_or(default_diarized);
            if let Some(current) = current_speaker.filter(|current| *current != speaker) {
                projected_chunks.push(PreparedMonologueChunk {
                    speaker: current.flatten(),
                    words: std::mem::take(&mut current_words),
                });
                stats.speaker_boundaries += 1;
            }
            current_speaker = Some(speaker);
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
        stats,
    }
}

fn project_timed_word(
    word: &AsrWord,
    segments: &[SpeakerSegment],
    label_coordinates: &DiarizationLabelCoordinates,
    stats: &mut SpeakerProjectionStats,
) -> Option<DiarizationSpeakerIndex> {
    let (Some(word_start), Some(word_end)) = (word.start_ms, word.end_ms) else {
        stats.unattested_timed_words += 1;
        return None;
    };
    if word_start < 0 || word_end <= word_start {
        stats.unattested_timed_words += 1;
        return None;
    }
    let (word_start, word_end) = (word_start as u64, word_end as u64);
    let mut overlap_by_speaker = vec![0u64; label_coordinates.len()];
    for segment in segments {
        let overlap_start = word_start.max(segment.interval.start_millis());
        let overlap_end = word_end.min(segment.interval.end_millis());
        if overlap_end > overlap_start {
            // Construction invariant: coordinates were built from every
            // segment in this same slice.
            #[allow(clippy::expect_used)]
            let index = label_coordinates
                .index_for(segment.speaker.as_str())
                .expect("segment label must be present in diarization coordinates")
                .0;
            overlap_by_speaker[index] += overlap_end - overlap_start;
        }
    }
    let positive_speakers = overlap_by_speaker
        .iter()
        .filter(|overlap| **overlap > 0)
        .count();
    if positive_speakers == 0 {
        stats.unattested_timed_words += 1;
        return segments
            .iter()
            .enumerate()
            .min_by_key(|(position, segment)| {
                let (start, end) = (
                    segment.interval.start_millis(),
                    segment.interval.end_millis(),
                );
                let distance = if word_end <= start {
                    start - word_end
                } else {
                    word_start.saturating_sub(end)
                };
                (distance, *position)
            })
            .and_then(|(_, segment)| label_coordinates.index_for(segment.speaker.as_str()));
    }
    if positive_speakers > 1 {
        stats.contested_timed_words += 1;
    }
    let (index, _) = overlap_by_speaker
        .iter()
        .enumerate()
        .max_by_key(|(index, overlap)| (**overlap, std::cmp::Reverse(*index)))?;
    Some(DiarizationSpeakerIndex(index))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn word(text: &str, start_ms: Option<i64>, end_ms: Option<i64>) -> AsrWord {
        AsrWord::new(text, start_ms, end_ms)
    }

    #[test]
    fn splits_at_diarization_boundary_and_attaches_punctuation_backward() {
        let chunks = vec![PreparedMonologueChunk {
            speaker: SpeakerIndex(0),
            words: vec![
                word("hello", Some(0), Some(500)),
                word("world", Some(500), Some(1_000)),
                word(".", None, None),
            ],
        }];
        let segments = vec![
            SpeakerSegment {
                interval: batchalign_types::interval::AdmittedInterval::admit_millis(0, 500)
                    .expect("an ordered fixture span"),
                speaker: "A".into(),
            },
            SpeakerSegment {
                interval: batchalign_types::interval::AdmittedInterval::admit_millis(500, 1_000)
                    .expect("an ordered fixture span"),
                speaker: "B".into(),
            },
        ];

        let projected = project_speakers_onto_chunks(chunks, &segments);

        assert_eq!(projected.chunks.len(), 2);
        assert_eq!(projected.chunks[0].speaker, SpeakerIndex(0));
        assert_eq!(projected.chunks[1].speaker, SpeakerIndex(1));
        assert_eq!(projected.chunks[1].words.len(), 2);
        assert_eq!(projected.stats.speaker_boundaries, 1);
    }

    #[test]
    fn speaker_indices_use_the_same_lexical_label_coordinates_as_turn_artifacts() {
        let chunks = vec![PreparedMonologueChunk {
            speaker: SpeakerIndex(0),
            words: vec![
                word("first", Some(0), Some(500)),
                word("second", Some(500), Some(1_000)),
            ],
        }];
        let segments = vec![
            SpeakerSegment {
                interval: batchalign_types::interval::AdmittedInterval::admit_millis(0, 500)
                    .expect("an ordered fixture span"),
                speaker: "SPEAKER_01".into(),
            },
            SpeakerSegment {
                interval: batchalign_types::interval::AdmittedInterval::admit_millis(500, 1_000)
                    .expect("an ordered fixture span"),
                speaker: "SPEAKER_00".into(),
            },
        ];

        let projected = project_speakers_onto_chunks(chunks, &segments);

        assert_eq!(projected.chunks[0].speaker, SpeakerIndex(1));
        assert_eq!(projected.chunks[1].speaker, SpeakerIndex(0));
    }

    #[test]
    fn uncovered_words_remain_in_the_dedicated_label_space() {
        let chunks = vec![PreparedMonologueChunk {
            speaker: SpeakerIndex(0),
            words: vec![
                word("hello", Some(0), Some(500)),
                word("later", Some(2_000), Some(2_500)),
            ],
        }];
        let segments = vec![SpeakerSegment {
            interval: batchalign_types::interval::AdmittedInterval::admit_millis(0, 500)
                .expect("an ordered fixture span"),
            speaker: "A".into(),
        }];

        let projected = project_speakers_onto_chunks(chunks, &segments);

        assert_eq!(projected.chunks.len(), 1);
        assert_eq!(projected.chunks[0].speaker, SpeakerIndex(0));
        assert_eq!(projected.stats.unattested_timed_words, 1);
    }

    #[test]
    fn gaps_choose_the_nearest_dedicated_segment_without_phantom_speakers() {
        let chunks = vec![PreparedMonologueChunk {
            speaker: SpeakerIndex(7),
            words: vec![
                word("left", Some(0), Some(200)),
                word("near_right", Some(850), Some(950)),
                word("right", Some(1_000), Some(1_200)),
            ],
        }];
        let segments = vec![
            SpeakerSegment {
                interval: batchalign_types::interval::AdmittedInterval::admit_millis(0, 200)
                    .expect("an ordered fixture span"),
                speaker: "A".into(),
            },
            SpeakerSegment {
                interval: batchalign_types::interval::AdmittedInterval::admit_millis(1_000, 1_200)
                    .expect("an ordered fixture span"),
                speaker: "B".into(),
            },
        ];

        let projected = project_speakers_onto_chunks(chunks, &segments);

        assert_eq!(projected.chunks.len(), 2);
        assert_eq!(projected.chunks[0].speaker, SpeakerIndex(0));
        assert_eq!(projected.chunks[1].speaker, SpeakerIndex(1));
        assert_eq!(projected.chunks[1].words[0].text, "near_right");
        assert_eq!(projected.stats.unattested_timed_words, 1);
        assert!(
            projected
                .chunks
                .iter()
                .all(|chunk| chunk.speaker.as_usize() < 2)
        );
    }

    #[test]
    fn reports_contested_words_and_uses_greatest_total_overlap() {
        let chunks = vec![PreparedMonologueChunk {
            speaker: SpeakerIndex(0),
            words: vec![word("hello", Some(0), Some(1_000))],
        }];
        let segments = vec![
            SpeakerSegment {
                interval: batchalign_types::interval::AdmittedInterval::admit_millis(0, 700)
                    .expect("an ordered fixture span"),
                speaker: "A".into(),
            },
            SpeakerSegment {
                interval: batchalign_types::interval::AdmittedInterval::admit_millis(600, 1_000)
                    .expect("an ordered fixture span"),
                speaker: "B".into(),
            },
        ];

        let projected = project_speakers_onto_chunks(chunks, &segments);

        assert_eq!(projected.chunks[0].speaker, SpeakerIndex(0));
        assert_eq!(projected.stats.contested_timed_words, 1);
    }

    #[test]
    fn empty_segments_preserve_chunks_exactly() {
        let chunks = vec![PreparedMonologueChunk {
            speaker: SpeakerIndex(4),
            words: vec![word("hello", Some(0), Some(500))],
        }];

        let projected = project_speakers_onto_chunks(chunks.clone(), &[]);

        assert_eq!(projected.chunks, chunks);
        assert_eq!(projected.stats, SpeakerProjectionStats::default());
    }
}
