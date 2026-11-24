//! The one conversion from recording milliseconds to prepared-audio frames.
//!
//! # The coordinate space this module adds, and why it is a third one
//!
//! `crate::time` already separates [`FileMs`], a position measured from the
//! start of the recording, from `WindowMs`, what an alignment engine reports
//! relative to the window it was handed. Speaker identification introduces a
//! third space: a FRAME INDEX into the prepared mono PCM view the worker
//! holds. It is a different space for the same reason the other two are: the
//! numbers are not interchangeable, and nothing but the sample rate and the
//! length of that particular decode relates them.
//!
//! The rule is the one `chat_ops::fa::coordinates` states for time:
//! [`PreparedPcm::locate`] is the ONLY route from a [`MediaWindow`] to a
//! [`FrameSpan`], because it is the only place that holds both the sample rate
//! and the frame count, and therefore the only place the containment question
//! can be asked. [`FrameSpan`] has no constructor from bare integers, so a
//! span cannot be minted from two numbers somebody computed by hand.
//!
//! # What `locate` refuses, and why that is a verdict rather than an error
//!
//! A bullet naming audio the recording does not contain is a real, recurring
//! state: transcripts outlive the media they were made against, and the FA
//! module exists partly because timings 28.2 seconds past the end of a file
//! reached real deliveries. Here it is refused, and the refusal carries the
//! bullet and the recording length so the caller can report the utterance
//! UNSCORED with the numbers a reader needs, rather than embedding whatever
//! bytes happen to sit at the end of the buffer.

use crate::media::window::MediaWindow;
use crate::time::Ms;

/// A half-open range of frames in one prepared PCM decode.
///
/// Constructible only by [`PreparedPcm::locate`]. Possession of one is proof
/// that the range lies inside the decode it was located in, which is the
/// pairing a bare `(u64, u64)` cannot express: two integers do not say which
/// buffer they index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct FrameSpan {
    start: u64,
    end: u64,
}

impl FrameSpan {
    /// First frame, inclusive.
    #[must_use]
    pub const fn start(self) -> u64 {
        self.start
    }

    /// Last frame, exclusive.
    #[must_use]
    pub const fn end(self) -> u64 {
        self.end
    }

    /// How many frames the span holds.
    #[must_use]
    pub const fn len(self) -> u64 {
        self.end - self.start
    }

    /// Whether the span holds no frames.
    ///
    /// Always false for a located span, because a [`MediaWindow`] is non-empty
    /// and a non-empty window of a positive sample rate is at least one frame.
    /// Present because clippy asks for it beside `len`, and because a reader
    /// should not have to reconstruct that argument.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.start == self.end
    }
}

/// A window that names audio the prepared decode does not contain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error(
    "[{start_ms}ms, {end_ms}ms) is not inside the prepared audio, which runs to {recording_ms}ms"
)]
pub struct OutsidePreparedAudio {
    /// Start of the requested window.
    pub start_ms: u64,
    /// End of the requested window.
    pub end_ms: u64,
    /// Length of the prepared decode.
    pub recording_ms: u64,
}

/// The prepared mono PCM view of one recording.
///
/// Owns the sample rate and the frame count together, so a caller cannot pair
/// one decode's rate with another decode's length. That pairing is exactly the
/// relationship the type exists to hold: two values maintained by convention
/// is the shape where a wrong combination type-checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PreparedPcm {
    sample_rate_hz: u32,
    frame_count: u64,
}

/// A prepared decode that cannot index anything.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum NotAPreparedDecode {
    /// A zero sample rate relates no milliseconds to any frame.
    #[error("prepared audio cannot have a zero sample rate")]
    ZeroSampleRate,
    /// A decode with no frames holds no audio to identify anyone by.
    #[error("prepared audio cannot be empty")]
    NoFrames,
}

impl PreparedPcm {
    /// Describe one prepared decode, refusing a rate or length that indexes
    /// nothing.
    pub const fn new(sample_rate_hz: u32, frame_count: u64) -> Result<Self, NotAPreparedDecode> {
        if sample_rate_hz == 0 {
            return Err(NotAPreparedDecode::ZeroSampleRate);
        }
        if frame_count == 0 {
            return Err(NotAPreparedDecode::NoFrames);
        }
        Ok(Self {
            sample_rate_hz,
            frame_count,
        })
    }

    /// The decode's sample rate, for the evidence's provenance block.
    #[must_use]
    pub const fn sample_rate_hz(self) -> u32 {
        self.sample_rate_hz
    }

    /// How long the decode runs.
    #[must_use]
    pub const fn duration(self) -> Ms {
        Ms(self.frame_count.saturating_mul(1000) / self.sample_rate_hz as u64)
    }

    /// The single transition from recording milliseconds to frames.
    ///
    /// Rounds the start DOWN and the end UP, so the located span never holds
    /// less audio than the window asked for. Rounding the other way would
    /// shave a few milliseconds off every utterance, which matters only at the
    /// model's minimum length, which is exactly where a wrong answer changes a
    /// verdict from measured to unmeasurable.
    pub fn locate(self, window: MediaWindow) -> Result<FrameSpan, OutsidePreparedAudio> {
        let rate = u64::from(self.sample_rate_hz);
        let start = window.start().get().saturating_mul(rate) / 1000;
        let end = window
            .end()
            .get()
            .saturating_mul(rate)
            .div_ceil(1000)
            .min(self.frame_count);

        let outside = || OutsidePreparedAudio {
            start_ms: window.start().get(),
            end_ms: window.end().get(),
            recording_ms: self.duration().0,
        };

        if start >= self.frame_count || end <= start {
            return Err(outside());
        }
        Ok(FrameSpan { start, end })
    }
}

#[cfg(test)]
#[allow(clippy::panic)]
mod tests {
    use super::*;
    use crate::time::FileMs;

    fn pcm() -> PreparedPcm {
        match PreparedPcm::new(16_000, 16_000 * 30) {
            Ok(pcm) => pcm,
            Err(error) => panic!("a legal decode: {error}"),
        }
    }

    fn window(start_ms: u64, end_ms: u64) -> MediaWindow {
        match MediaWindow::new(FileMs::new(start_ms), FileMs::new(end_ms)) {
            Ok(window) => window,
            Err(error) => panic!("a legal window: {error}"),
        }
    }

    /// MEASUREMENT: the millisecond-to-frame arithmetic, at the rate the
    /// prepared decode actually uses.
    #[test]
    fn milliseconds_locate_at_the_decodes_own_rate() {
        let span = match pcm().locate(window(1_000, 2_000)) {
            Ok(span) => span,
            Err(error) => panic!("inside the recording: {error}"),
        };
        assert_eq!(span.start(), 16_000);
        assert_eq!(span.end(), 32_000);
        assert_eq!(span.len(), 16_000);
    }

    /// The span never holds LESS than the window asked for: the end rounds up.
    /// A shaved millisecond only matters at the model's minimum length, which
    /// is precisely where it would flip a verdict to unmeasurable.
    #[test]
    fn a_fractional_end_rounds_up_so_no_audio_is_lost() {
        let span = match pcm().locate(window(0, 1)) {
            Ok(span) => span,
            Err(error) => panic!("inside the recording: {error}"),
        };
        assert_eq!(span.len(), 16);
    }

    /// A window past the end is REFUSED with the numbers a reader needs, never
    /// silently shortened to whatever bytes remain in the buffer.
    #[test]
    fn a_window_past_the_end_is_refused_with_the_recording_length() {
        match pcm().locate(window(31_000, 32_000)) {
            Err(OutsidePreparedAudio {
                start_ms,
                end_ms,
                recording_ms,
            }) => {
                assert_eq!(start_ms, 31_000);
                assert_eq!(end_ms, 32_000);
                assert_eq!(recording_ms, 30_000);
            }
            Ok(span) => panic!("expected a refusal, got {span:?}"),
        }
    }

    /// A window straddling the end is clipped to the decode rather than
    /// refused: the audio it does cover is real, and refusing every final
    /// utterance because a bullet overruns by a few milliseconds would make
    /// the command useless on transcripts that end at the recording's edge.
    #[test]
    fn a_window_straddling_the_end_keeps_the_audio_that_exists() {
        let span = match pcm().locate(window(29_500, 31_000)) {
            Ok(span) => span,
            Err(error) => panic!("partially inside the recording: {error}"),
        };
        assert_eq!(span.end(), 16_000 * 30);
        assert!(!span.is_empty());
    }

    #[test]
    fn a_decode_that_indexes_nothing_is_refused() {
        assert_eq!(
            PreparedPcm::new(0, 100),
            Err(NotAPreparedDecode::ZeroSampleRate)
        );
        assert_eq!(
            PreparedPcm::new(16_000, 0),
            Err(NotAPreparedDecode::NoFrames)
        );
    }
}
