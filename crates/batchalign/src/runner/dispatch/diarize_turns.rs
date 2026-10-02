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

use serde::Serialize;

use crate::chat_ops::speaker::DiarizationLabelCoordinates;
use crate::types::worker_v2::{SpeakerBackendV2, SpeakerSegmentV2};

/// Provenance recorded in a speaker-turn artifact.
///
/// The worker backend is a closed enum, and the artifact source is derived
/// from that enum rather than supplied as a free-form string. This prevents a
/// cloud Precision-2 result from being mislabeled as local pyannote output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SpeakerTurnsSource {
    /// pyannoteAI cloud diarization using the Precision-2 model.
    PyannoteAiPrecision2,
    /// Local pyannote diarization.
    Pyannote,
    /// Local NVIDIA NeMo diarization.
    Nemo,
}

impl SpeakerTurnsSource {
    /// Derive artifact provenance from the backend that produced the turns.
    pub(crate) fn from_backend(backend: SpeakerBackendV2) -> Self {
        match backend {
            SpeakerBackendV2::PyannoteAi => Self::PyannoteAiPrecision2,
            SpeakerBackendV2::Pyannote => Self::Pyannote,
            SpeakerBackendV2::Nemo => Self::Nemo,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::PyannoteAiPrecision2 => "batchalign3:pyannote_ai:precision-2",
            Self::Pyannote => "batchalign3:pyannote",
            Self::Nemo => "batchalign3:nemo",
        }
    }
}

/// Anonymous CHAT-style speaker track (`PAR0`..`PARn`).
///
/// A track is an acoustic identity assigned by the diarizer, NOT a CHAT
/// role; role assignment is downstream work (`chatter rediarize` /
/// speaker-id). The inner index is the track number.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct AnonymousTrack(pub(crate) usize);

impl Serialize for AnonymousTrack {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(&format_args!("PAR{}", self.0))
    }
}

/// One diarized speaker turn: an anonymous track plus its media span.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub(crate) struct DiarizedTurn {
    /// Millisecond media offsets first so the derived `Ord` sorts turns
    /// chronologically (start, end, track), the order chatter expects.
    pub(crate) start_ms: u64,
    /// Exclusive end of the turn in media milliseconds.
    pub(crate) end_ms: u64,
    /// Anonymous track code for the voice heard in this span.
    pub(crate) track: AnonymousTrack,
}

/// The complete turns artifact for one media file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct DiarizedTurnsFile {
    /// Engine provenance (`batchalign3:pyannote`).
    pub(crate) source: &'static str,
    /// All detected turns, chronologically ordered.
    pub(crate) turns: Vec<DiarizedTurn>,
}

/// Errors while converting worker speaker segments to the turns artifact.
#[derive(Debug, thiserror::Error)]
pub(crate) enum TurnsBuildError {
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
pub(crate) fn format_turns_json(
    source: SpeakerTurnsSource,
    segments: &[SpeakerSegmentV2],
) -> Result<String, TurnsBuildError> {
    let label_coordinates = DiarizationLabelCoordinates::from_labels(
        segments.iter().map(|segment| segment.speaker.as_str()),
    );

    let mut turns = Vec::with_capacity(segments.len());
    for segment in segments {
        // Ordered by its type: the segment's interval was admitted at parse.
        let (start_ms, end_ms) = (
            segment.interval.start_millis(),
            segment.interval.end_millis(),
        );
        // Map-lookup invariant: every segment label was inserted into
        // `track_by_label` by the collection pass above.
        #[allow(clippy::expect_used)]
        let track = label_coordinates
            .index_for(segment.speaker.as_str())
            .expect("segment label must be present in the label->track map");
        turns.push(DiarizedTurn {
            start_ms,
            end_ms,
            track: AnonymousTrack(track.as_usize()),
        });
    }
    turns.sort();

    let file = DiarizedTurnsFile {
        source: source.as_str(),
        turns,
    };
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

    #[test]
    fn maps_sorted_labels_to_par_tracks_and_sorts_turns() {
        let segments = [
            segment("SPEAKER_01", 5000, 6000),
            segment("SPEAKER_00", 0, 1200),
            segment("SPEAKER_01", 1200, 2000),
        ];
        let json =
            format_turns_json(SpeakerTurnsSource::Pyannote, &segments).expect("valid segments");
        let value: serde_json::Value = serde_json::from_str(&json).expect("well-formed JSON");
        assert_eq!(value["source"], "batchalign3:pyannote");
        let turns = value["turns"].as_array().expect("turns array");
        // Chronological order, SPEAKER_00 -> PAR0, SPEAKER_01 -> PAR1.
        assert_eq!(turns[0]["track"], "PAR0");
        assert_eq!(turns[0]["start_ms"], 0);
        assert_eq!(turns[1]["track"], "PAR1");
        assert_eq!(turns[1]["start_ms"], 1200);
        assert_eq!(turns[2]["end_ms"], 6000);
    }

    #[test]
    fn empty_segments_produce_an_empty_turns_document() {
        let json =
            format_turns_json(SpeakerTurnsSource::Pyannote, &[]).expect("empty input is valid");
        let value: serde_json::Value = serde_json::from_str(&json).expect("well-formed JSON");
        assert_eq!(value["turns"].as_array().map(Vec::len), Some(0));
    }

    /// An inverted worker segment never reaches turn building: the segment
    /// is refused where it is parsed.
    #[test]
    fn an_inverted_segment_is_refused_where_it_is_parsed() {
        let refused = serde_json::from_str::<SpeakerSegmentV2>(
            r#"{"start_ms":2000,"end_ms":1000,"speaker":"SPEAKER_00"}"#,
        );
        assert!(refused.is_err(), "{refused:?}");
    }

    #[test]
    fn wire_shape_matches_the_chatter_rediarize_contract_exactly() {
        // chatter's TurnsFile parser is strict; pin the exact field set.
        let json = format_turns_json(
            SpeakerTurnsSource::Pyannote,
            &[segment("SPEAKER_00", 10, 20)],
        )
        .expect("valid");
        let value: serde_json::Value = serde_json::from_str(&json).expect("well-formed JSON");
        let object = value.as_object().expect("top-level object");
        assert_eq!(
            object.keys().collect::<Vec<_>>(),
            ["source", "turns"],
            "top-level field set drifted from the chatter contract"
        );
        let turn = value["turns"][0].as_object().expect("turn object");
        let mut keys = turn.keys().collect::<Vec<_>>();
        keys.sort();
        assert_eq!(
            keys,
            ["end_ms", "start_ms", "track"],
            "turn field set drifted from the chatter contract"
        );
    }

    #[test]
    fn derives_truthful_source_from_every_typed_speaker_backend() {
        let cases = [
            (
                SpeakerBackendV2::PyannoteAi,
                "batchalign3:pyannote_ai:precision-2",
            ),
            (SpeakerBackendV2::Pyannote, "batchalign3:pyannote"),
            (SpeakerBackendV2::Nemo, "batchalign3:nemo"),
        ];

        for (backend, expected) in cases {
            let json = format_turns_json(
                SpeakerTurnsSource::from_backend(backend),
                &[segment("SPEAKER_00", 10, 20)],
            )
            .expect("valid");
            let value: serde_json::Value = serde_json::from_str(&json).expect("well-formed JSON");
            assert_eq!(value["source"], expected);
        }
    }
}
