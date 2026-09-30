//! The second question, asked only when the first could not be trusted: how
//! long do the audio stream's packets actually run?
//!
//! # What is summed, and why not a packet count
//!
//! ffprobe reports every packet's duration in the stream's time base. Summing
//! them needs no table of samples per frame (1152 for MPEG-1 Layer III, 576 for
//! MPEG-2 and 2.5, 384 for Layer I, 1024 for AAC) and no assumption that the
//! bitrate or even the sample rate is constant: the demuxer's parser read each
//! frame header and says what that frame holds. A count multiplied by one
//! frame size would restate a table ffmpeg already owns, and be wrong for the
//! first stream that switched.
//!
//! From the sum it subtracts the samples the decoder TRIMS, which ffprobe
//! reports as `skip_samples` (encoder delay at the start) and
//! `discard_padding` (encoder padding at the end) on the packets that carry
//! them. LAME writes both into its Info header; without the subtraction a
//! walked 44.1 kHz MP3 runs about 29 ms longer than its decode, and an 8 kHz
//! one about 210 ms longer. With it, the result equals the decoded sample
//! count to the millisecond (measured 2026-09-30, ffmpeg 9.0.2, against a full
//! decode: MPEG-1 at 44.1 kHz, MPEG-2 at 22.05 kHz, MPEG-2.5 at 11.025 and
//! 8 kHz, CBR and VBR, with and without Info headers; ADTS AAC; ADPCM WAV).
//!
//! # Cost
//!
//! A walk demuxes the whole file without decoding it: every byte is read once.
//! ffprobe's JSON for a 2.5-hour MP3 (340,000 packets) is about 11 MB. The
//! bytes are buffered (ffprobe's output is collected like every media tool's),
//! but the packets are summed as they are deserialized, never collected into a
//! list of 340,000 values.
//!
//! # What a walk cannot see
//!
//! Bytes the demuxer skips while resynchronizing (garbage between frames)
//! never become packets, so they are absent from the walk AND from the decode,
//! which agree. The walk measures the timeline a decoder produces, which is
//! what a bound needs; it does not detect a gap in the file's own timeline.

use std::fmt;
use std::num::{NonZeroU32, NonZeroU64};

use serde::Deserialize;
use serde::de::{Deserializer, SeqAccess, Visitor};

use super::ProbeError;
use super::duration::{AudioDuration, DurationBasis, WalkReason};
use crate::api::DurationMs;

/// ffprobe's answer to the walk question.
#[derive(Debug, Deserialize)]
pub(super) struct WalkAnswer {
    packets: PacketTotals,
    streams: Vec<WalkStream>,
}

/// The stream's clock: the unit packet durations are in, and the rate the
/// trimmed-sample counts are in.
#[derive(Debug, Deserialize)]
struct WalkStream {
    /// `"1/14112000"`.
    time_base: String,
    /// `"44100"`.
    sample_rate: String,
}

/// Everything the walk needs from the packets, accumulated during
/// deserialization.
#[derive(Debug, Default)]
struct PacketTotals {
    /// Sum of packet durations, in time-base ticks.
    ticks: u128,
    /// Samples the decoder drops: start skip plus end padding.
    trimmed_samples: u128,
    /// Packets ffprobe reported without a duration. Any at all makes the sum
    /// a lower bound rather than a length, so it is refused, not ignored.
    without_duration: u64,
}

/// One packet, as asked for.
#[derive(Debug, Deserialize)]
struct Packet {
    /// Absent when the demuxer could not say.
    duration: Option<u64>,
    /// Present on the few packets that carry side data.
    side_data_list: Option<Vec<SideDatum>>,
}

/// One side-data entry. Only the skip-samples kind carries these keys; the
/// selection prints any other kind as an empty entry, which contributes
/// nothing because it is not a trim.
#[derive(Debug, Deserialize)]
struct SideDatum {
    skip_samples: Option<u64>,
    discard_padding: Option<u64>,
}

impl<'de> Deserialize<'de> for PacketTotals {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Summing;

        impl<'de> Visitor<'de> for Summing {
            type Value = PacketTotals;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("ffprobe's packet array")
            }

            fn visit_seq<A: SeqAccess<'de>>(self, mut packets: A) -> Result<Self::Value, A::Error> {
                let mut totals = PacketTotals::default();
                while let Some(packet) = packets.next_element::<Packet>()? {
                    match packet.duration {
                        Some(ticks) => totals.ticks += u128::from(ticks),
                        None => totals.without_duration += 1,
                    }
                    for datum in packet.side_data_list.iter().flatten() {
                        totals.trimmed_samples += u128::from(datum.skip_samples.unwrap_or(0))
                            + u128::from(datum.discard_padding.unwrap_or(0));
                    }
                }
                Ok(totals)
            }
        }

        deserializer.deserialize_seq(Summing)
    }
}

/// A time base, `numerator / denominator` seconds per tick.
#[derive(Clone, Copy, Debug)]
struct TimeBase {
    numerator: NonZeroU64,
    denominator: NonZeroU64,
}

impl TimeBase {
    /// ffprobe's `"n/d"`.
    fn parse(text: &str) -> Option<Self> {
        let (numerator, denominator) = text.split_once('/')?;
        Some(Self {
            numerator: numerator.parse().ok()?,
            denominator: denominator.parse().ok()?,
        })
    }
}

impl WalkAnswer {
    /// The walked length, rounded up to the millisecond.
    ///
    /// Exact rational arithmetic in 128 bits: `ticks * n / d` seconds of
    /// packets, less `trimmed / rate` seconds of trims. Ten hours at a
    /// 1/14,112,000 time base is about 5e11 ticks; scaled by a 48 kHz rate
    /// and 1000 it is still eleven orders of magnitude inside `u128`.
    pub(super) fn length(
        self,
        input: &str,
        reason: WalkReason,
    ) -> Result<AudioDuration, ProbeError> {
        let unreadable = |detail: String| ProbeError::Unreadable {
            input: input.to_owned(),
            detail,
        };
        let stream = self
            .streams
            .into_iter()
            .next()
            .ok_or_else(|| ProbeError::NoAudioStream {
                input: input.to_owned(),
            })?;
        let time_base = TimeBase::parse(&stream.time_base)
            .ok_or_else(|| unreadable(format!("time base {:?}", stream.time_base)))?;
        let rate: NonZeroU32 = stream
            .sample_rate
            .parse()
            .map_err(|_| unreadable(format!("sample rate {:?}", stream.sample_rate)))?;
        match self.packets.without_duration {
            0 => {}
            count => {
                return Err(ProbeError::PacketsWithoutDuration {
                    input: input.to_owned(),
                    count,
                });
            }
        }

        let rate = u128::from(rate.get());
        let numerator = u128::from(time_base.numerator.get());
        let denominator = u128::from(time_base.denominator.get());
        // Both terms over the common denominator `denominator * rate`, in ms.
        // Checked, because the counts are ffprobe's and a nonsensical answer
        // must be refused rather than wrap or panic.
        let overflow = || unreadable("packet totals overflow 128 bits".to_owned());
        let product = |factors: [u128; 4]| {
            factors
                .into_iter()
                .try_fold(1u128, u128::checked_mul)
                .ok_or_else(overflow)
        };
        let packets = product([self.packets.ticks, numerator, rate, 1000])?;
        let trims = product([self.packets.trimmed_samples, denominator, 1000, 1])?;
        let net = packets.checked_sub(trims).ok_or_else(|| {
            unreadable(format!(
                "the decoder would trim {} samples from a stream shorter than that",
                self.packets.trimmed_samples
            ))
        })?;
        let milliseconds = net.div_ceil(product([denominator, rate, 1, 1])?);
        let milliseconds = u64::try_from(milliseconds)
            .map_err(|_| unreadable(format!("a length of {milliseconds} ms")))?;
        AudioDuration::established(DurationMs(milliseconds), DurationBasis::Walked(reason))
            .ok_or_else(|| ProbeError::EmptyAudio {
                input: input.to_owned(),
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn walked(json: &str) -> Result<AudioDuration, ProbeError> {
        serde_json::from_str::<WalkAnswer>(json)
            .expect("well-formed answer")
            .length("x", WalkReason::MpegAudio)
    }

    /// The arithmetic on ffprobe's own spelling, for an MP3 with an Info
    /// header: 1429 frames of 1152 samples, less LAME's 1105-sample delay and
    /// 173 samples of padding, is 1,644,930 samples, which is what a decode of
    /// that file produced (37.300 s).
    #[test]
    fn trims_are_subtracted_from_the_packet_sum() {
        let middle = r#"{ "duration": 368640 },"#.repeat(1427);
        let json = format!(
            r#"{{ "packets": [
                {{ "duration": 368640, "side_data_list": [ {{ "skip_samples": 1105, "discard_padding": 0 }} ] }},
                {middle}
                {{ "duration": 368640, "side_data_list": [ {{ "skip_samples": 0, "discard_padding": 173 }} ] }}
              ],
              "streams": [ {{ "sample_rate": "44100", "time_base": "1/14112000" }} ] }}"#
        );
        let duration = walked(&json).expect("walks");
        assert_eq!(duration.length(), DurationMs(37_300));
        assert_eq!(
            duration.basis(),
            DurationBasis::Walked(WalkReason::MpegAudio)
        );
    }

    /// Side data of another kind prints as an empty entry and is not a trim.
    #[test]
    fn side_data_that_is_not_a_trim_contributes_nothing() {
        let json = r#"{ "packets": [ { "duration": 1000, "side_data_list": [ { } ] } ],
                        "streams": [ { "sample_rate": "1000", "time_base": "1/1000" } ] }"#;
        assert_eq!(walked(json).expect("walks").length(), DurationMs(1_000));
    }

    /// A fractional final millisecond rounds up, so the bound covers it.
    #[test]
    fn a_fractional_length_rounds_up() {
        // 3 packets of 1152 samples at 44.1 kHz: 78.367 ms.
        let json = r#"{ "packets": [ { "duration": 1152 }, { "duration": 1152 }, { "duration": 1152 } ],
                        "streams": [ { "sample_rate": "44100", "time_base": "1/44100" } ] }"#;
        assert_eq!(walked(json).expect("walks").length(), DurationMs(79));
    }

    /// A packet whose duration nobody knows makes the sum a lower bound, which
    /// is exactly the defect this module exists to remove, so it is refused.
    #[test]
    fn a_packet_without_a_duration_is_refused() {
        let json = r#"{ "packets": [ { "duration": 1000 }, { } ],
                        "streams": [ { "sample_rate": "1000", "time_base": "1/1000" } ] }"#;
        assert!(matches!(
            walked(json),
            Err(ProbeError::PacketsWithoutDuration { count: 1, .. })
        ));
    }

    /// A stream with no packets holds no audio, refused where it is born.
    #[test]
    fn a_walk_of_no_packets_is_empty_audio() {
        let json =
            r#"{ "packets": [], "streams": [ { "sample_rate": "1000", "time_base": "1/1000" } ] }"#;
        assert!(matches!(walked(json), Err(ProbeError::EmptyAudio { .. })));
    }

    #[test]
    fn trims_longer_than_the_stream_are_unreadable() {
        let json = r#"{ "packets": [ { "duration": 10, "side_data_list": [ { "skip_samples": 11, "discard_padding": 0 } ] } ],
                        "streams": [ { "sample_rate": "1000", "time_base": "1/1000" } ] }"#;
        assert!(matches!(walked(json), Err(ProbeError::Unreadable { .. })));
    }
}
