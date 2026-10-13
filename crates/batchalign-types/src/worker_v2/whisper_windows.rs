//! Window ownership retained before a provider can merge repeated text.

use serde::{Deserialize, Serialize};

use super::requests::{FrameCountV2, SampleRateHzV2, WhisperChunkSpanV2};

/// Unwindowed producers keep their original wire shape. A windowed producer
/// cannot attach ownership beside a second, already-merged transcript.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, schemars::JsonSchema)]
#[serde(untagged)]
pub enum WhisperObservationsV2 {
    /// Native or legacy observations with no claimed window ownership.
    Monolithic(Vec<WhisperChunkSpanV2>),
    /// Independently decoded, sample-owned windows.
    Windows(WhisperWindowsV2),
}

impl From<Vec<WhisperChunkSpanV2>> for WhisperObservationsV2 {
    fn from(spans: Vec<WhisperChunkSpanV2>) -> Self {
        Self::Monolithic(spans)
    }
}

/// One independent decode. Timestamps are local to this window, not the file.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct WhisperWindowV2 {
    /// First source sample observed by this decode.
    pub start_sample: FrameCountV2,
    /// Number of source samples, excluding model padding.
    pub sample_count: FrameCountV2,
    /// Left context shared with the preceding decode.
    pub left_context: FrameCountV2,
    /// Right context shared with the following decode.
    pub right_context: FrameCountV2,
    /// Raw word or segment observations, never cross-window merged.
    pub spans: Vec<WhisperChunkSpanV2>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
struct WindowWire {
    sample_rate_hz: SampleRateHzV2,
    windows: Vec<WhisperWindowV2>,
}

/// A contiguous sequence with nonempty owned interiors. Deserialization is
/// admission: consumers cannot obtain a sequence with fabricated gaps or seams.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, schemars::JsonSchema)]
#[serde(try_from = "WindowWire", into = "WindowWire")]
pub struct WhisperWindowsV2(WindowWire);

/// The worker supplied inconsistent sample ownership.
#[derive(Debug, Clone, thiserror::Error)]
#[error("invalid Whisper window ownership: {0}")]
pub struct WhisperWindowRefusal(&'static str);

impl TryFrom<WindowWire> for WhisperWindowsV2 {
    type Error = WhisperWindowRefusal;

    fn try_from(wire: WindowWire) -> Result<Self, Self::Error> {
        let refuse = WhisperWindowRefusal;
        if wire.sample_rate_hz.0 == 0 || wire.windows.is_empty() {
            return Err(refuse("a sample rate and at least one window are required"));
        }
        let mut owned_end = 0_u64;
        for window in &wire.windows {
            let contexts = window
                .left_context
                .0
                .checked_add(window.right_context.0)
                .ok_or(refuse("context overflow"))?;
            if contexts >= window.sample_count.0 {
                return Err(refuse("window has no owned interior"));
            }
            let owned_start = window
                .start_sample
                .0
                .checked_add(window.left_context.0)
                .ok_or(refuse("start overflow"))?;
            if owned_start != owned_end {
                return Err(refuse(
                    "owned interiors must be contiguous and start at zero",
                ));
            }
            owned_end = window
                .start_sample
                .0
                .checked_add(window.sample_count.0)
                .and_then(|end| end.checked_sub(window.right_context.0))
                .ok_or(refuse("end overflow"))?;
        }
        if wire.windows.first().is_some_and(|w| w.left_context.0 != 0)
            || wire.windows.last().is_some_and(|w| w.right_context.0 != 0)
        {
            return Err(refuse("outer windows cannot claim exterior context"));
        }
        Ok(Self(wire))
    }
}

impl From<WhisperWindowsV2> for WindowWire {
    fn from(value: WhisperWindowsV2) -> Self {
        value.0
    }
}

impl WhisperWindowsV2 {
    /// Source sampling rate shared by every observation.
    pub fn sample_rate_hz(&self) -> SampleRateHzV2 {
        self.0.sample_rate_hz
    }
    /// Admitted windows in source order.
    pub fn windows(&self) -> &[WhisperWindowV2] {
        &self.0.windows
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn window_ownership_is_admitted_at_deserialization() {
        let valid = serde_json::json!({"sample_rate_hz": 16000, "windows": [
            {"start_sample":0,"sample_count":400000,"left_context":0,
             "right_context":48000,"spans":[]},
            {"start_sample":304000,"sample_count":200000,"left_context":48000,
             "right_context":0,"spans":[]}
        ]});
        let admitted: WhisperWindowsV2 = serde_json::from_value(valid.clone()).expect("admitted");
        assert_eq!(serde_json::to_value(admitted).expect("wire"), valid);
        for (field, value) in [
            ("start_sample", 304001),
            ("sample_count", 48000),
            ("right_context", 1),
            ("left_context", 0),
        ] {
            let mut invalid = valid.clone();
            invalid["windows"][1][field] = value.into();
            assert!(
                serde_json::from_value::<WhisperWindowsV2>(invalid).is_err(),
                "{field}"
            );
        }
        let mut invalid = valid;
        invalid["sample_rate_hz"] = 0.into();
        assert!(serde_json::from_value::<WhisperWindowsV2>(invalid).is_err());
        assert!(
            serde_json::from_value::<WhisperWindowsV2>(
                serde_json::json!({"sample_rate_hz":16000,"windows":[]})
            )
            .is_err()
        );
    }
}
