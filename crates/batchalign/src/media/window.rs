//! A window of a media file, which cannot be empty.
//!
//! Its own module because it is a MEDIA primitive, not a transcode detail: the
//! CHAT-analysis side produces windows (`chat_ops::fa::find_untimed_windows`)
//! and the transcode side consumes them, so putting it under either would make
//! the other depend on a neighbour it has no business knowing.
//!
//! # Why it exists
//!
//! The same `end <= start` comparison was written in THREE places, none of them
//! where the window originates: `validate_fa_infer_item` checked it, then
//! `extract_prepared_audio_segment_f32le` checked it again on the same numbers,
//! and `extract_audio_segment` checked it a third time for the UTR path. The
//! producer, `find_untimed_windows`, returned bare `(u64, u64)` tuples and
//! checked nothing, while being the one place that could actually build an
//! inverted one.

use std::ffi::OsString;
use std::num::NonZeroU64;

use crate::time::FileMs;

/// A non-empty half-open window of a source file.
///
/// Constructible only when `end` follows `start`, so a caller holding one has
/// already proved the window can contain audio. `artifacts_v2` used to check
/// this inline and report the failure as `io::ErrorKind::InvalidInput`, which
/// described an invalid ARGUMENT as a failure of the filesystem.
/// The bounds are [`FileMs`] because they are POSITIONS, not lengths; see
/// [`crate::time`] for why that distinction has its own module.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MediaWindow {
    start: FileMs,
    end: FileMs,
}

/// A window whose end does not follow its start.
///
/// `Copy` and comparable because it is two positions and nothing else, and
/// because callers wrap it as the `#[source]` of their own errors, which then
/// cannot derive `Clone`/`PartialEq` unless this one does. An error type that
/// cannot be compared is one a test can only assert the shape of.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("media window end {end} must be greater than start {start}")]
pub struct EmptyWindow {
    /// Requested start.
    pub start: FileMs,
    /// Requested end.
    pub end: FileMs,
}

impl MediaWindow {
    /// The window from `start` to `end`, or [`EmptyWindow`] if it holds nothing.
    pub fn new(start: FileMs, end: FileMs) -> Result<Self, EmptyWindow> {
        if end <= start {
            return Err(EmptyWindow { start, end });
        }
        Ok(Self { start, end })
    }

    /// Start, as the position it is.
    ///
    /// There is no `start_ms() -> u64` beside this. There was, and adding
    /// `start()` without removing it meant the change added two accessors and
    /// removed nothing, leaving the untyped route out with the shorter name so
    /// a new call site would reach for it. Callers that genuinely need the
    /// integer (a cache key, a `Display`, tracing fields) spell the lowering
    /// themselves with `.get()`, which is one visible step rather than a second
    /// sanctioned way to ask the same question.
    #[must_use]
    pub fn start(self) -> FileMs {
        self.start
    }

    /// End, as the position it is.
    #[must_use]
    pub fn end(self) -> FileMs {
        self.end
    }

    /// `ffmpeg`'s seconds-with-milliseconds spelling of this window.
    pub(in crate::media) fn as_seek_args(self) -> [OsString; 4] {
        [
            OsString::from("-ss"),
            OsString::from(format!("{:.3}", self.start.get() as f64 / 1000.0)),
            OsString::from("-to"),
            OsString::from(format!("{:.3}", self.end.get() as f64 / 1000.0)),
        ]
    }
}

/// Why a decode of a [`MediaWindow`] produced no whole sample frames.
///
/// # Why it is a sum and not a comment
///
/// `ServerError::EmptyFaAudioSegment` carried the window and the path and
/// nothing else, so the reason a segment was empty could only be guessed by
/// whoever read the log line. Two places guessed, in writing, and both guessed
/// the same unprovable thing: `error.rs` said the window "falls past the end of
/// the source audio file", and `fa::transport`'s own handler contradicts that
/// three lines into its warning ("that does not prove the window is past EOF;
/// very short in-range windows can do this").
///
/// So there is deliberately NO `PastEndOfAudio` variant. The party that
/// measures holds a byte length and no source duration, and a reason it cannot
/// witness is exactly the guess this type exists to stop.
///
/// # Every route to one
///
/// [`DecodedFrames::measure_f32le`] and nothing else, and since 2026-09-07 that
/// is ENFORCED rather than asserted: the facts live in a private enum, so no
/// caller outside this module can name one. The sentence above was already
/// false when it was written, because the variants and their payload were
/// public and a `#[cfg(test)]` caller in `worker::artifacts_v2` was building
/// the struct literal for a reason nothing had measured. A reason a caller can
/// assert is not evidence, it is a label, and this one decides what an operator
/// is told went wrong.
///
/// # The cross-product, and the two cells that are gone
///
/// The public payload admitted `byte_len == 0`, which is `NoBytesDecoded` under
/// another name, and `byte_len >= sample_bytes`, which is at least one whole
/// frame and therefore not an emptiness at all. Neither can come from the real
/// producer and neither has a constructor now: [`PartialFrame`] is built by one
/// fallible private constructor that refuses both, so even this module cannot
/// state a partial frame that is not one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EmptyReason(Emptiness);

/// What a decode measurement saw, kept private so the reason travels as
/// EVIDENCE and not as an assertion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Emptiness {
    /// The decoder wrote NOTHING for this window.
    ///
    /// `ffmpeg` exits 0 and produces an empty file when the window yields no
    /// audio, so a successful exit is not evidence that audio exists. What it
    /// does NOT say is why the window yielded nothing; see above.
    NoBytesDecoded,
    /// The decoder wrote bytes, but fewer than one whole sample.
    ///
    /// A DIFFERENT operational fact, and one nothing could say before: the
    /// frame count was `byte_len / sample_bytes`, integer division, so a
    /// truncated write (a killed transcode, a full disk) rounded down to zero
    /// and was reported as an empty window. That sends an operator to look at
    /// the transcript's timings instead of at the machine.
    PartialFrame(PartialFrame),
}

/// A decode that wrote bytes without completing one sample.
///
/// Both invariants live in the one constructor rather than in a comment: the
/// byte count is non-zero because zero bytes is the OTHER emptiness, and it is
/// short of a whole sample because a whole sample would have been a frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PartialFrame {
    /// Bytes the decoder actually wrote.
    byte_len: NonZeroU64,
    /// Bytes in one whole sample of this encoding.
    sample_bytes: NonZeroU64,
}

impl PartialFrame {
    /// The partial frame `byte_len` bytes make, if they make one at all.
    ///
    /// `None` for zero bytes and for a count that completes a sample, so the
    /// caller has to say which of the two facts it actually measured.
    fn shorter_than_one_sample(byte_len: u64, sample_bytes: u64) -> Option<Self> {
        let byte_len = NonZeroU64::new(byte_len)?;
        let sample_bytes = NonZeroU64::new(sample_bytes)?;
        (byte_len < sample_bytes).then_some(Self {
            byte_len,
            sample_bytes,
        })
    }
}

impl std::fmt::Display for EmptyReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0 {
            Emptiness::NoBytesDecoded => write!(f, "the decoder produced no bytes"),
            Emptiness::PartialFrame(PartialFrame {
                byte_len,
                sample_bytes,
            }) => write!(
                f,
                "the decoder produced {byte_len} bytes, less than one {sample_bytes}-byte sample"
            ),
        }
    }
}

/// How many whole sample frames a decode produced.
///
/// The success arm cannot hold zero, and the empty arm carries WHY. Before
/// this, `artifacts_v2` divided, compared the quotient to zero, and built its
/// error from the window alone: the reason was discarded at the one place that
/// could observe it, and a `FrameCountV2(0)` was one forgotten `if` away from
/// reaching a worker, where an empty tensor crashes the model with an opaque
/// kernel-size error.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodedFrames {
    /// At least one whole sample frame was decoded.
    Frames(std::num::NonZeroU64),
    /// No whole sample frame was decoded, and this is what the measurement saw.
    None(EmptyReason),
}

impl DecodedFrames {
    /// Measure a raw 32-bit-float PCM decode, in bytes.
    ///
    /// The sample width is this function's OWN knowledge rather than a
    /// parameter, so there is no frame size for a caller to get wrong and no
    /// zero divisor to defend against. `artifacts_v2` spelled
    /// `size_of::<f32>()` inline beside its own `PcmEncoding::F32LeRaw`; the
    /// two agreed, and nothing said they had to.
    #[must_use]
    pub fn measure_f32le(byte_len: u64) -> Self {
        const SAMPLE_BYTES: u64 = size_of::<f32>() as u64;
        match NonZeroU64::new(byte_len / SAMPLE_BYTES) {
            Some(frames) => Self::Frames(frames),
            // The division has already established that there is no whole
            // sample here, so the constructor's `None` is exactly the
            // zero-byte case and not a fallback.
            None => Self::None(EmptyReason(
                match PartialFrame::shorter_than_one_sample(byte_len, SAMPLE_BYTES) {
                    Some(partial) => Emptiness::PartialFrame(partial),
                    None => Emptiness::NoBytesDecoded,
                },
            )),
        }
    }
}

/// A media window that yielded no audio.
///
/// `ffmpeg` exits 0 and writes an empty or truncated output rather than
/// failing, so a successful exit is not evidence that audio exists, and this is
/// the one failure such an exit can report. WHY the window yielded nothing is
/// not something the exit status says, and this value does not guess: it
/// carries the [`EmptyReason`] the measuring party actually observed. The
/// sentence here claimed the window "falls past the end of the source, so this
/// is the one failure a SUCCESSFUL exit reports" until 2026-09-07, which is the
/// same unwitnessable cause `EmptyReason` was added to replace and which had
/// already been removed from `error.rs` in the same change.
///
/// It has to travel: the data plane detects it, the request builder turns it
/// into a skip, and the transport layer acts on it three layers up.
///
/// # Why it is a type
///
/// It was `{ path: String, start_ms, end_ms }` declared in THREE error enums,
/// with a field-by-field rebuild at each layer boundary whose only job was to
/// move three values between structurally identical variants. The two loose
/// integers also disagreed about their own type: `u64` in the two worker enums
/// and `DurationMs` in `ServerError`, converted in passing by one of the
/// rebuilds. Carrying the [`MediaWindow`] that was ASKED FOR removes both
/// problems: one declaration, and the window keeps the type it was proven at.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EmptySegment {
    /// Source media the window was requested from.
    pub path: String,
    /// The window that produced nothing.
    pub window: MediaWindow,
    /// What the measurement saw, issued by the party that measured it.
    ///
    /// Added 2026-09-07. Without it the value said a segment was empty and
    /// could not say anything more, so every consumer that wanted a reason
    /// invented one; see [`EmptyReason`] for the two that did, in writing.
    pub reason: EmptyReason,
}

impl std::fmt::Display for EmptySegment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{}ms..{}ms) in {}: {}",
            self.window.start().get(),
            self.window.end().get(),
            self.path,
            self.reason
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A window cannot be empty, so "a zero-length window" is not one of the
    /// reasons a decode can come back with nothing: it is unrepresentable.
    ///
    /// Kept because it is the argument for why [`EmptyReason`] has the
    /// variants it has, and that argument lives in a type rather than in the
    /// enum's prose.
    #[test]
    fn a_window_that_holds_nothing_cannot_be_built() {
        let at = FileMs::new(1_000);
        assert!(MediaWindow::new(at, at).is_err());
        assert!(MediaWindow::new(FileMs::new(2_000), FileMs::new(1_000)).is_err());
        assert!(MediaWindow::new(FileMs::new(1_000), FileMs::new(1_001)).is_ok());
    }

    /// RED FIRST (2026-09-07): a decode that wrote SOME bytes but fewer than
    /// one whole sample is a different fact from one that wrote none, and the
    /// measuring party is the only one that can tell them apart.
    ///
    /// `byte_len / sample_bytes` is integer division, so before this the two
    /// cases were the same value and a truncated write was reported as an
    /// empty window.
    #[test]
    fn a_partial_sample_is_not_the_same_emptiness_as_no_bytes() {
        // Rendered rather than compared against a literal, because a literal
        // is exactly what no caller can write any more: the two facts are
        // observable through what they SAY, which is what an operator reads.
        assert_eq!(
            reason_of(DecodedFrames::measure_f32le(0)).to_string(),
            "the decoder produced no bytes"
        );
        assert_eq!(
            reason_of(DecodedFrames::measure_f32le(3)).to_string(),
            "the decoder produced 3 bytes, less than one 4-byte sample",
            "a truncated decode must say so rather than reporting an empty window"
        );
        assert_ne!(
            reason_of(DecodedFrames::measure_f32le(0)),
            reason_of(DecodedFrames::measure_f32le(3)),
            "the two emptinesses are different facts, not one message"
        );
    }

    /// The reason a decode of `byte_len` bytes reports, from the measurement.
    ///
    /// The only way to obtain an [`EmptyReason`] anywhere, tests included.
    fn reason_of(measured: DecodedFrames) -> EmptyReason {
        match measured {
            DecodedFrames::None(reason) => reason,
            DecodedFrames::Frames(frames) => {
                panic!("expected an empty decode, got {frames} whole frames")
            }
        }
    }

    /// The success arm counts whole samples and cannot be zero, so a caller
    /// that matched `Frames` has a frame count it can hand to a model.
    #[test]
    fn measuring_counts_whole_samples_and_discards_the_remainder_as_frames() {
        let frames = |byte_len| match DecodedFrames::measure_f32le(byte_len) {
            DecodedFrames::Frames(frames) => Some(frames.get()),
            DecodedFrames::None(_) => None,
        };
        assert_eq!(frames(4), Some(1));
        assert_eq!(frames(8), Some(2));
        // A trailing partial sample does not make the decode empty; the whole
        // samples before it are real.
        assert_eq!(frames(9), Some(2));
    }
}
