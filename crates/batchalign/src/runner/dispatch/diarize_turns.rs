//! Typed model for the `diarize` command's speaker-turns JSON artifact.
//!
//! Contract: the emitted document is exactly the input schema consumed by
//! `chatter rediarize --turns` (talkbank-transform's `TurnsFile`):
//!
//! ```json
//! {"source": "batchalign3:pyannote",
//!  "turns": [{"track": "PAR0", "start_ms": 0, "end_ms": 1200}]}
//! ```
//!
//! chatter parses the file strictly (unknown or misspelled fields fail the
//! parse rather than being dropped), so the field names here are
//! load-bearing wire format, not internal naming.

use batchalign_types::interval::AdmittedInterval;
use serde::Serialize;

use crate::chat_ops::speaker::DiarizationLabelCoordinates;
use crate::types::worker_v2::{SpeakerBackendV2, SpeakerSegmentV2};

/// Provenance recorded in a speaker-turn artifact.
///
/// The worker backend is a closed enum, and the artifact source is derived
/// from that enum rather than supplied as a free-form string. This prevents a
/// cloud Precision-2 result from being mislabeled as local pyannote output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum SpeakerTurnsSource {
    /// pyannoteAI cloud diarization using the Precision-2 model.
    #[serde(rename = "batchalign3:pyannote_ai:precision-2")]
    PyannoteAiPrecision2,
    /// Local pyannote diarization.
    #[serde(rename = "batchalign3:pyannote")]
    Pyannote,
}

impl SpeakerTurnsSource {
    /// Derive artifact provenance from the backend that produced the turns.
    pub(crate) const fn from_backend(backend: SpeakerBackendV2) -> Self {
        match backend {
            SpeakerBackendV2::PyannoteAi => Self::PyannoteAiPrecision2,
            SpeakerBackendV2::Pyannote => Self::Pyannote,
        }
    }
}

/// Anonymous CHAT-style speaker track (`PAR0`..`PARn`).
///
/// A track is an acoustic identity assigned by the diarizer, NOT a CHAT
/// role; role assignment is downstream work (`chatter rediarize` /
/// speaker-id). The inner index is the track number.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct AnonymousTrack(usize);

impl Serialize for AnonymousTrack {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(&format_args!("PAR{}", self.0))
    }
}

/// One diarized speaker turn: an anonymous track plus its media span.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
struct DiarizedTurn {
    // The wire still has start_ms and end_ms, while ordering uses the
    // original admitted worker interval followed by the track.
    #[serde(flatten)]
    interval: AdmittedInterval,
    track: AnonymousTrack,
}

/// The complete turns artifact for one media file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct DiarizedTurnsFile {
    source: SpeakerTurnsSource,
    turns: Vec<DiarizedTurn>,
}

/// Errors while converting worker speaker segments to the turns artifact.
#[derive(Debug, thiserror::Error)]
pub enum TurnsBuildError {
    /// The typed model failed to serialize (indicates a programming error,
    /// surfaced rather than panicking per the no-panic policy).
    #[error("failed to serialize turns JSON: {0}")]
    Serialize(#[from] serde_json::Error),
}

/// Convert raw worker segments into the canonical turns artifact text.
///
/// Diarizer-native labels (pyannote's `SPEAKER_00`, ...) map to anonymous
/// track codes deterministically: distinct labels sorted lexically become
/// `PAR0..PARn`. Turns are emitted in chronological order.
pub fn format_turns_json(
    source: SpeakerTurnsSource,
    segments: &[SpeakerSegmentV2],
) -> Result<String, TurnsBuildError> {
    let mut turns: Vec<_> = DiarizationLabelCoordinates::numbered_turns(segments)
        .map(|(track, interval)| DiarizedTurn {
            interval,
            track: AnonymousTrack(track.as_usize()),
        })
        .collect();
    turns.sort();

    let file = DiarizedTurnsFile { source, turns };
    let mut text = serde_json::to_string_pretty(&file)?;
    text.push('\n');
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn segment(speaker: &str, start_ms: i64, end_ms: i64) -> SpeakerSegmentV2 {
        SpeakerSegmentV2 {
            interval: batchalign_types::interval::AdmittedInterval::admit_millis(start_ms, end_ms)
                .expect("an ordered fixture"),
            speaker: speaker.to_owned(),
        }
    }

    /// Guards: Real turns JSON keeps exact Chatter fields, reports its actual
    /// backend, and orders admitted intervals under lexical labels.
    /// Kind: wire
    /// Not a type: The receiving Chatter parser uses externally spelled JSON
    /// fields, track codes and backend strings that Rust names independently.
    /// State: Three in-memory worker intervals and both supported sources.
    /// Concurrency: Serialization reads only the test's admitted values.
    #[test]
    fn wire_shape_matches_the_chatter_rediarize_contract_exactly() {
        let segments = [
            segment("SPEAKER_01", 5000, 6000),
            segment("SPEAKER_00", 0, 1200),
            segment("SPEAKER_01", 1200, 2000),
        ];
        for (backend, expected_source) in [
            (SpeakerBackendV2::Pyannote, "batchalign3:pyannote"),
            (
                SpeakerBackendV2::PyannoteAi,
                "batchalign3:pyannote_ai:precision-2",
            ),
        ] {
            let source = SpeakerTurnsSource::from_backend(backend);
            let json = format_turns_json(source, &segments).expect("turns");
            let value: serde_json::Value = serde_json::from_str(&json).expect("JSON");
            assert_eq!(
                value,
                serde_json::json!({
                    "source": expected_source,
                    "turns": [
                        { "track": "PAR0", "start_ms": 0, "end_ms": 1200 },
                        { "track": "PAR1", "start_ms": 1200, "end_ms": 2000 },
                        { "track": "PAR1", "start_ms": 5000, "end_ms": 6000 }
                    ]
                }),
            );
        }
        let empty = format_turns_json(SpeakerTurnsSource::Pyannote, &[]).expect("empty turns");
        let value: serde_json::Value = serde_json::from_str(&empty).expect("JSON");
        assert_eq!(
            value,
            serde_json::json!({ "source": "batchalign3:pyannote", "turns": [] })
        );
    }

    /// Guards: The worker wire refuses inverted offsets before producing a segment.
    /// Kind: wire
    /// Not a type: An external peer can send inverted millisecond values
    /// before the shared interval type admits them.
    /// State: Exact raw worker JSON exercises the DTO's flattened interval.
    /// Concurrency: No process, ambient inputs or mutable data.
    #[test]
    fn an_inverted_segment_is_refused_where_it_is_parsed() {
        let refused = serde_json::from_str::<SpeakerSegmentV2>(
            r#"{"start_ms":2000,"end_ms":1000,"speaker":"SPEAKER_00"}"#,
        );
        assert!(refused.is_err(), "{refused:?}");
    }
}
