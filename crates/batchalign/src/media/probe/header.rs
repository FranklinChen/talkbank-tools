//! The first question: what container is this, and does it state its length?
//!
//! A header read, cheap on any file. Its answer decides the route: take the
//! stated length ([`DurationRoute::Stated`]) or walk the packets
//! ([`DurationRoute::Walk`]). The decision is a table over a closed set of
//! demuxers, so a container nobody has measured is refused rather than
//! trusted.

use serde::Deserialize;

use super::ProbeError;
use super::duration::{AudioDuration, DurationBasis, StatingContainer, WalkReason};
use crate::api::DurationMs;

/// ffprobe's answer to the header question, as its JSON writer spells it.
///
/// `streams` is not defaulted: ffprobe always writes the array, empty when the
/// file has no audio stream, so a missing key is an unreadable answer rather
/// than "no audio".
#[derive(Debug, Deserialize)]
pub(super) struct HeaderAnswer {
    streams: Vec<HeaderStream>,
    format: HeaderFormat,
}

/// The first audio stream, which is the one every decode in this crate reads.
#[derive(Debug, Deserialize)]
struct HeaderStream {
    codec_name: String,
    /// The stream's own stated length, absent when the container states none
    /// per stream (Matroska).
    duration: Option<String>,
}

/// The container-level entries asked for.
#[derive(Debug, Deserialize)]
struct HeaderFormat {
    /// The demuxer's name. A comma-separated list for demuxers that serve a
    /// family (`mov,mp4,m4a,3gp,3g2,mj2`), matched whole.
    format_name: String,
    /// The file's length, which is the LONGEST stream's: a video longer than
    /// its audio makes this the video's. Absent, not zero, when the container
    /// states none: the absence is the fact.
    duration: Option<String>,
}

/// Which way the length must be established.
#[derive(Debug)]
pub(super) enum DurationRoute {
    /// The container stated it, and the container is one whose statement is
    /// exact. Nothing more to read.
    Stated(AudioDuration),
    /// The stated length cannot be trusted, or is absent; walk the packets.
    Walk(WalkReason),
}

/// What the header says to do with a file, before its stated length is read.
enum Classified {
    /// The demuxer's stated length is an estimate or a claim; walk.
    Walk(WalkReason),
    /// The demuxer states an exact length, or states none.
    Stating(StatingContainer),
}

/// Where a stating container's length is stated.
enum Statement {
    /// Per stream, so the audio's own length is read.
    AudioStream,
    /// Only for the whole file (the longest stream's length).
    File,
}

/// What to do when a stating container states nothing.
enum Unstated {
    /// Its packets sum exactly (measured), so walk them.
    Walk,
    /// Its packets do not (measured 0.3 s short), so neither route is exact.
    Refuse,
}

/// The routing policy for each stating container, next to the table in
/// [`HeaderAnswer::route`]. Exhaustive, so a new container must take a side on
/// both questions.
impl StatingContainer {
    fn statement(self) -> Statement {
        match self {
            Self::PcmWav | Self::Flac | Self::IsoMedia | Self::Ogg => Statement::AudioStream,
            Self::Matroska | Self::Asf => Statement::File,
        }
    }

    fn when_unstated(self) -> Unstated {
        match self {
            Self::PcmWav | Self::Flac => Unstated::Walk,
            Self::IsoMedia | Self::Ogg | Self::Matroska | Self::Asf => Unstated::Refuse,
        }
    }
}

/// The measured demuxers, by ffprobe's `format_name` matched whole, and the
/// first audio stream's codec where the demuxer alone does not decide. The
/// catch-all is over ffprobe's open vocabulary: an unmeasured demuxer is
/// refused by name, never trusted.
fn classify(format_name: &str, codec_name: &str) -> Option<Classified> {
    match format_name {
        "mp3" => Some(Classified::Walk(WalkReason::MpegAudio)),
        "aac" => Some(Classified::Walk(WalkReason::Adts)),
        // Every PCM codec ffmpeg knows is spelled `pcm_*`; a WAV holding
        // anything else has an estimated length.
        "wav" => Some(match codec_name.starts_with("pcm_") {
            true => Classified::Stating(StatingContainer::PcmWav),
            false => Classified::Walk(WalkReason::CompressedWav),
        }),
        "flac" => Some(Classified::Stating(StatingContainer::Flac)),
        "mov,mp4,m4a,3gp,3g2,mj2" => Some(Classified::Stating(StatingContainer::IsoMedia)),
        "ogg" => Some(Classified::Stating(StatingContainer::Ogg)),
        "matroska,webm" => Some(Classified::Stating(StatingContainer::Matroska)),
        "asf" => Some(Classified::Stating(StatingContainer::Asf)),
        _ => None,
    }
}

impl HeaderAnswer {
    /// Decide the route for this file.
    ///
    /// The table is the policy, and every row was measured (see
    /// [`StatingContainer`] and [`WalkReason`]):
    ///
    /// | demuxer | stated length | route |
    /// |---|---|---|
    /// | MPEG audio, ADTS | any | walk |
    /// | WAV with a compressed codec | any | walk |
    /// | WAV with PCM, FLAC | present | stated |
    /// | WAV with PCM, FLAC | absent | walk |
    /// | ISO media, Ogg, Matroska, ASF | present | stated |
    /// | ISO media, Ogg, Matroska, ASF | absent | refused |
    /// | anything else | any | refused |
    ///
    /// A walked container's stated length is never parsed, so whatever
    /// ffprobe printed for it cannot fail the probe.
    pub(super) fn route(self, input: &str) -> Result<DurationRoute, ProbeError> {
        let Self { streams, format } = self;
        let stream = streams
            .into_iter()
            .next()
            .ok_or_else(|| ProbeError::NoAudioStream {
                input: input.to_owned(),
            })?;
        let container = match classify(&format.format_name, &stream.codec_name) {
            Some(Classified::Walk(reason)) => return Ok(DurationRoute::Walk(reason)),
            Some(Classified::Stating(container)) => container,
            None => {
                return Err(ProbeError::UnmeasuredContainer {
                    input: input.to_owned(),
                    demuxer: format.format_name,
                });
            }
        };
        let stated = match container.statement() {
            Statement::AudioStream => stream.duration,
            Statement::File => format.duration,
        };
        match (stated, container.when_unstated()) {
            (Some(answer), _) => {
                let length = stated_length(input, &answer)?;
                AudioDuration::established(length, DurationBasis::Stated(container))
                    .map(DurationRoute::Stated)
                    .ok_or_else(|| ProbeError::EmptyAudio {
                        input: input.to_owned(),
                    })
            }
            (None, Unstated::Walk) => Ok(DurationRoute::Walk(WalkReason::LengthNotStated)),
            (None, Unstated::Refuse) => Err(ProbeError::LengthNotStated {
                input: input.to_owned(),
                container,
            }),
        }
    }
}

/// A stated length in ffprobe's decimal seconds, rounded UP to the millisecond.
///
/// ffprobe prints microseconds (`37.300000`). The seconds are brought to whole
/// microseconds before rounding up, because `37.3 * 1000.0` is not exactly
/// 37300 in binary floating point and a direct ceiling would add a millisecond
/// the container never stated.
fn stated_length(input: &str, answer: &str) -> Result<DurationMs, ProbeError> {
    let unreadable = || ProbeError::Unreadable {
        input: input.to_owned(),
        detail: format!("stated duration {answer:?} is not a non-negative number of seconds"),
    };
    let seconds: f64 = answer.parse().map_err(|_| unreadable())?;
    // Bounded above so the cast below cannot saturate into a fabricated
    // length.
    match seconds.is_finite() && (0.0..=u64::MAX as f64 / 1_000_000.0).contains(&seconds) {
        true => {
            let micros = (seconds * 1_000_000.0).round() as u64;
            Ok(DurationMs(micros.div_ceil(1000)))
        }
        false => Err(unreadable()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An answer whose audio stream and whole file state the same length, or
    /// none: the shape of an audio-only file.
    fn answer(format_name: &str, codec: &str, duration: Option<&str>) -> HeaderAnswer {
        answer_with(format_name, codec, duration, duration)
    }

    fn answer_with(
        format_name: &str,
        codec: &str,
        stream_duration: Option<&str>,
        file_duration: Option<&str>,
    ) -> HeaderAnswer {
        HeaderAnswer {
            streams: vec![HeaderStream {
                codec_name: codec.to_owned(),
                duration: stream_duration.map(str::to_owned),
            }],
            format: HeaderFormat {
                format_name: format_name.to_owned(),
                duration: file_duration.map(str::to_owned),
            },
        }
    }

    fn route(format_name: &str, codec: &str, duration: Option<&str>) -> DurationRoute {
        answer(format_name, codec, duration)
            .route("x")
            .expect("routable")
    }

    /// The routing table, row by row. POLICY: each row is a measured choice
    /// with alternatives, which is what makes it worth pinning.
    #[test]
    fn every_measured_demuxer_takes_its_route() {
        // MPEG audio is walked even when its header states a length: with a
        // Xing header the number is an encoder's claim, without one it is an
        // estimate, and ffprobe does not say which it printed.
        assert!(matches!(
            route("mp3", "mp3", Some("8855.073188")),
            DurationRoute::Walk(WalkReason::MpegAudio)
        ));
        assert!(matches!(
            route("aac", "aac", Some("40.847")),
            DurationRoute::Walk(WalkReason::Adts)
        ));
        assert!(matches!(
            route("wav", "adpcm_ima_wav", Some("10.0")),
            DurationRoute::Walk(WalkReason::CompressedWav)
        ));
        assert!(matches!(
            route("flac", "flac", None),
            DurationRoute::Walk(WalkReason::LengthNotStated)
        ));
        for (demuxer, codec, container) in [
            ("wav", "pcm_s16le", StatingContainer::PcmWav),
            ("flac", "flac", StatingContainer::Flac),
            ("mov,mp4,m4a,3gp,3g2,mj2", "aac", StatingContainer::IsoMedia),
            ("ogg", "opus", StatingContainer::Ogg),
            ("matroska,webm", "opus", StatingContainer::Matroska),
            ("asf", "wmav2", StatingContainer::Asf),
        ] {
            match route(demuxer, codec, Some("37.300000")) {
                DurationRoute::Stated(duration) => {
                    assert_eq!(duration.length(), DurationMs(37_300), "{demuxer}");
                    assert_eq!(duration.basis(), DurationBasis::Stated(container));
                }
                DurationRoute::Walk(reason) => panic!("{demuxer} walked for {reason:?}"),
            }
        }
    }

    /// A container that states each stream's length is measured by its AUDIO
    /// stream's, not the file's: an MP4 whose video runs past its audio
    /// (ffprobe measured: file 10.0 s, audio stream 6.0 s) is 6 s of audio.
    #[test]
    fn a_video_longer_than_its_audio_does_not_lengthen_the_audio() {
        let video = answer_with(
            "mov,mp4,m4a,3gp,3g2,mj2",
            "aac",
            Some("6.000000"),
            Some("10.0"),
        );
        match video.route("x") {
            Ok(DurationRoute::Stated(duration)) => assert_eq!(duration.length(), DurationMs(6_000)),
            other => panic!("expected the audio stream's length, got {other:?}"),
        }
    }

    /// A zero length is refused where it is born, so no recording is empty.
    #[test]
    fn a_stated_length_of_zero_is_empty_audio() {
        assert!(matches!(
            answer("wav", "pcm_s16le", Some("0.000000")).route("x"),
            Err(ProbeError::EmptyAudio { .. })
        ));
    }

    /// A container whose packets cannot be trusted to sum, and which states no
    /// length, is refused: neither route is exact for it.
    #[test]
    fn a_stating_container_that_states_nothing_is_refused() {
        let refused = answer("matroska,webm", "aac", None).route("x");
        assert!(matches!(
            refused,
            Err(ProbeError::LengthNotStated {
                container: StatingContainer::Matroska,
                ..
            })
        ));
    }

    /// A demuxer nobody has measured is refused by name, never trusted.
    #[test]
    fn an_unmeasured_demuxer_is_refused_by_name() {
        match answer("aiff", "pcm_s16be", Some("1.0")).route("x") {
            Err(ProbeError::UnmeasuredContainer { demuxer, .. }) => assert_eq!(demuxer, "aiff"),
            other => panic!("expected a refusal, got {other:?}"),
        }
    }

    #[test]
    fn a_file_without_audio_is_refused() {
        let silent = HeaderAnswer {
            streams: Vec::new(),
            format: HeaderFormat {
                format_name: "mp3".to_owned(),
                duration: Some("1.0".to_owned()),
            },
        };
        assert!(matches!(
            silent.route("x"),
            Err(ProbeError::NoAudioStream { .. })
        ));
    }

    /// Rounding is UP, from whole microseconds, so a stated length that is a
    /// whole number of milliseconds is not pushed a millisecond long by binary
    /// floating point, and a fractional one covers its last instant.
    #[test]
    fn stated_lengths_round_up_from_whole_microseconds() {
        assert_eq!(stated_length("x", "37.300000").unwrap(), DurationMs(37_300));
        assert_eq!(stated_length("x", "37.300001").unwrap(), DurationMs(37_301));
        assert_eq!(stated_length("x", "0.000000").unwrap(), DurationMs(0));
        assert!(stated_length("x", "N/A").is_err());
        assert!(stated_length("x", "-1.0").is_err());
        assert!(stated_length("x", "inf").is_err());
        assert!(stated_length("x", "1e300").is_err(), "no saturated length");
    }
}
