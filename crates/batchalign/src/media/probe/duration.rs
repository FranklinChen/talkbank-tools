//! How long a recording's audio runs, and how that was established.
//!
//! # Why the length carries its basis
//!
//! The probe used to return a bare [`DurationMs`] read from ffprobe's
//! `format=duration`, and that one number meant two different things depending
//! on the file. For a WAV, a FLAC or an MP4 it is a count the container states:
//! exact. For an MP3 without a Xing header, or an ADTS AAC stream, the demuxer
//! has no count to read and ESTIMATES the length from the file size over the
//! declared bitrate. On constant-bitrate MP3s whose frames are never padded
//! (so the stream runs at 127.7 kbit/s while every header says 128), that
//! estimate is 0.23% short: 30 seconds on a four-hour recording. Every
//! consumer then treated the end of the estimate as the end of the recording,
//! discarding the ASR tokens and refusing the FA timings of the audio beyond it.
//!
//! Nothing in the type told a consumer which of the two it held, so nothing
//! could. [`AudioDuration`] is the answer: its constructor is private to the
//! probe, the probe never builds one from an estimate, and the value records
//! which of the two admissible routes produced it.

use std::num::NonZeroU64;

use crate::api::DurationMs;

/// The length of a recording's audio, established by a route that is exact for
/// its container, and never zero.
///
/// Constructible only inside `media::probe`, and only by the two routes
/// [`DurationBasis`] names, so possession is the proof: no consumer can hold a
/// bitrate estimate by accident, because no code path produces one of these
/// from it.
///
/// The length is rounded UP to the whole millisecond. It is used as a BOUND
/// (the last instant of the recording), and an instant the audio contains must
/// never fall outside it; rounding down would put the final fraction of a
/// millisecond past the end.
///
/// Zero is refused where the value is born (the probe reports
/// [`super::ProbeError::EmptyAudio`]), so every recording built from one is
/// non-empty and [`crate::chat_ops::fa::coordinates::Recording::of_audio`]
/// cannot fail.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AudioDuration {
    length: NonZeroU64,
    basis: DurationBasis,
}

impl AudioDuration {
    /// The one constructor, reachable only from the probe's two routes.
    /// `None` for a zero length, which the caller reports as empty audio.
    pub(super) fn established(length: DurationMs, basis: DurationBasis) -> Option<Self> {
        NonZeroU64::new(length.0).map(|length| Self { length, basis })
    }

    /// How long the audio runs.
    #[must_use]
    pub const fn length(self) -> DurationMs {
        DurationMs(self.length.get())
    }

    /// How the length was established, for provenance in logs and errors.
    #[must_use]
    pub const fn basis(self) -> DurationBasis {
        self.basis
    }
}

/// How an [`AudioDuration`] was established.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DurationBasis {
    /// Stated by a container whose stated length is a count its demuxer reads
    /// (samples, sample-table entries, a final granule position), never an
    /// estimate. Reading it costs one header read.
    Stated(StatingContainer),
    /// Measured by walking every packet of the audio stream and summing their
    /// durations, net of the samples the decoder trims (encoder delay and end
    /// padding). This is the length a full decode produces. It reads the whole
    /// file, so it is used only where the container's own number cannot be
    /// trusted.
    Walked(WalkReason),
}

/// A container whose stated length is admissible as the recording's duration.
///
/// WAV, FLAC, ISO media and Ogg state the AUDIO STREAM's own length. Matroska
/// and ASF state only the file's (the longest stream's): exact for an
/// audio-only file, and an upper bound on the audio when video runs longer,
/// which is the harmless direction for a bound.
///
/// Admitted by measurement, not by reputation: each was compared against the
/// length a full ffmpeg decode produces (2026-09-30, ffmpeg 9.0.2). WAV, FLAC
/// and the ISO media family agreed to the millisecond; Ogg and Matroska within
/// 10 ms (Opus pre-skip); ASF was 46 ms long. None of them estimates, and a
/// bound a few milliseconds long is harmless where one seconds short discards
/// speech. `stated_lengths_agree_with_a_full_decode` in this module's tests
/// repeats the comparison on every host that runs them.
///
/// The packet walk is deliberately NOT the fallback for these. The same
/// comparison found it 0.3 s short on AAC in Matroska and on WMA in ASF, whose
/// packets do not carry every sample's duration; a container here that states
/// no length is refused rather than walked.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StatingContainer {
    /// RIFF WAVE holding PCM: the length is the data size over a fixed rate.
    PcmWav,
    /// FLAC whose STREAMINFO states its total sample count.
    Flac,
    /// The ISO base media family (MP4, M4A, MOV): the sample tables.
    IsoMedia,
    /// Ogg: the final page's granule position.
    Ogg,
    /// Matroska and WebM: the segment's Duration element.
    Matroska,
    /// ASF (WMA): the file properties' play duration.
    Asf,
}

/// Why a recording's length had to be measured rather than read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WalkReason {
    /// MPEG audio (MP3, MP2). Without a Xing/Info header the demuxer estimates
    /// from the first frame's bitrate, which is wrong for unpadded CBR and for
    /// every VBR stream; with one, the header's frame count is a claim written
    /// by an encoder, which a truncated or concatenated file contradicts.
    MpegAudio,
    /// ADTS AAC, which has no length field at all and is always estimated.
    Adts,
    /// A WAV holding a compressed codec, whose length the demuxer estimates.
    CompressedWav,
    /// A lossless container that normally states its length but does not here
    /// (a FLAC or WAV written to a pipe, which cannot seek back to fill it in).
    LengthNotStated,
}
