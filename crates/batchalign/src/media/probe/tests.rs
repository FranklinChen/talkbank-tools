//! The probe at its real boundary: real media, generated here, read by the
//! real ffprobe, checked against what a real ffmpeg decode produces.
//!
//! Every test REQUIRES ffmpeg and ffprobe and fails without them; a skipped
//! boundary test reads as a pass in a log nobody opens. What a decode produces
//! is a property of the ffmpeg RELEASE, so these run against the one release
//! `scripts/ffmpeg-pin.sh` names; the gate refuses any other before they run.

use std::path::Path;

use super::fixtures::{
    ESTIMATED_LENGTH_MS, FRAMES, TRUE_LENGTH_MS, write_unpadded_cbr_mp3,
    write_unpadded_cbr_mpeg2_mp3,
};
use super::*;
use crate::api::DurationMs;

/// ffprobe's own `format=duration` for `path`, the number the probe returned
/// before it had two questions: the premise of the defect tests, checked
/// rather than assumed.
fn container_duration_ms(path: &Path) -> f64 {
    let output = MediaTool::Ffprobe
        .run([
            "-v".as_ref(),
            "error".as_ref(),
            "-show_entries".as_ref(),
            "format=duration".as_ref(),
            "-of".as_ref(),
            "csv=p=0".as_ref(),
            path.as_os_str(),
        ])
        .expect("test requires ffprobe");
    let seconds: f64 = String::from_utf8_lossy(&output.stdout)
        .trim()
        .parse()
        .expect("ffprobe states a duration");
    seconds * 1000.0
}

/// The length a full ffmpeg decode of `path` produces, in whole milliseconds
/// rounded up: the timeline every engine sees. Asserts the decode was clean,
/// so a fixture that only LOOKS valid cannot pass.
fn decoded_ms(path: &Path, rate_hz: u64) -> u64 {
    let output = MediaTool::Ffmpeg
        .command()
        .args(["-nostdin", "-v", "error", "-i"])
        .arg(path)
        .args(["-ac", "1", "-f", "s16le", "-acodec", "pcm_s16le", "-"])
        .output()
        .expect("test requires ffmpeg");
    assert!(
        output.status.success(),
        "decode of {} failed",
        path.display()
    );
    assert!(
        output.stderr.is_empty(),
        "decode of {} reported: {}",
        path.display(),
        String::from_utf8_lossy(&output.stderr)
    );
    let samples = output.stdout.len() as u64 / 2;
    (samples * 1000).div_ceil(rate_hz)
}

/// The length of a generated tone: the CONTENT a fixture holds, which is the
/// same on every host, unlike what a given ffmpeg's decoder returns for it.
#[derive(Clone, Copy)]
struct ToneMs(u64);

impl ToneMs {
    /// The tone most fixtures encode: long enough that a 0.23% estimate error
    /// is tens of milliseconds, and not a whole number of codec frames.
    const LONG: Self = Self(20_300);

    /// The lavfi source that generates this tone at 44.1 kHz.
    fn source(self) -> String {
        format!("sine=frequency=440:sample_rate=44100:duration={}ms", self.0)
    }
}

/// Encode a 440 Hz tone at 44.1 kHz with ffmpeg into `path`.
fn encode(path: &Path, tone: ToneMs, codec_args: &[&str]) {
    let status = MediaTool::Ffmpeg
        .command()
        .args(["-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i"])
        .arg(tone.source())
        .args(codec_args)
        .arg(path)
        .status()
        .expect("test requires ffmpeg");
    assert!(status.success(), "could not encode {}", path.display());
}

/// THE DEFECT, at the ffprobe boundary: an unpadded CBR MP3 without a Xing
/// header, whose container duration ffprobe ESTIMATES from the declared
/// bitrate and so under-reports by 0.23%. The probe must return the length
/// the frames hold, which is the length a decode produces.
#[tokio::test]
async fn an_unpadded_cbr_mp3_is_walked_not_estimated() {
    let dir = tempfile::tempdir().expect("tempdir");
    let source = dir.path().join("unpadded.mp3");
    write_unpadded_cbr_mp3(&source);

    let estimate = container_duration_ms(&source);
    assert!(
        (estimate - ESTIMATED_LENGTH_MS).abs() < 1.0,
        "premise: ffprobe estimates {estimate} ms from the declared bitrate"
    );
    assert_eq!(decoded_ms(&source, 44_100), TRUE_LENGTH_MS);

    let probed = MediaProbe::new(&source)
        .duration()
        .await
        .expect("the fixture probes");
    assert_eq!(probed.length(), DurationMs(TRUE_LENGTH_MS));
    assert_eq!(probed.basis(), DurationBasis::Walked(WalkReason::MpegAudio));
}

/// MPEG-2 Layer III holds 576 samples per frame, not 1152. The walk sums
/// what each packet holds rather than multiplying a count by a frame size, so
/// it needs no table to be right here.
#[tokio::test]
async fn an_unpadded_mpeg2_mp3_is_walked_at_576_samples_per_frame() {
    let dir = tempfile::tempdir().expect("tempdir");
    let source = dir.path().join("mpeg2.mp3");
    write_unpadded_cbr_mpeg2_mp3(&source);

    let exact = (FRAMES * 576 * 1000).div_ceil(22_050);
    assert!(
        container_duration_ms(&source) < exact as f64 - 500.0,
        "premise: the container estimate is short"
    );
    assert_eq!(decoded_ms(&source, 22_050), exact);
    let probed = MediaProbe::new(&source)
        .duration()
        .await
        .expect("the fixture probes");
    assert_eq!(probed.length(), DurationMs(exact));
}

/// An ffmpeg-encoded VBR MP3 without an Info header: the estimate comes from
/// the first frame's bitrate and misses by seconds; the walk does not.
#[tokio::test]
async fn a_vbr_mp3_without_an_info_header_is_walked() {
    let dir = tempfile::tempdir().expect("tempdir");
    let source = dir.path().join("vbr.mp3");
    encode(
        &source,
        ToneMs::LONG,
        &["-acodec", "libmp3lame", "-q:a", "2", "-write_xing", "0"],
    );
    let decoded = decoded_ms(&source, 44_100);
    assert!(
        (container_duration_ms(&source) - decoded as f64).abs() > 1_000.0,
        "premise: the estimate is more than a second off"
    );
    let probed = MediaProbe::new(&source).duration().await.expect("probes");
    assert_eq!(probed.length(), DurationMs(decoded));
}

/// An MP3 WITH a LAME Info header carries encoder delay and padding that the
/// decoder trims. The walk subtracts them, so it matches the decode exactly
/// rather than running a frame long; the blocking and async spawns agree.
#[tokio::test]
async fn trimmed_priming_is_not_counted_as_audio() {
    let dir = tempfile::tempdir().expect("tempdir");
    let source = dir.path().join("lame.mp3");
    encode(
        &source,
        ToneMs::LONG,
        &["-acodec", "libmp3lame", "-b:a", "128k"],
    );
    let decoded = decoded_ms(&source, 44_100);
    let probed = MediaProbe::new(&source).duration().await.expect("probes");
    assert_eq!(probed.length(), DurationMs(decoded));
    let blocking = MediaProbe::new(&source)
        .duration_blocking()
        .expect("probes");
    assert_eq!(blocking, probed);
}

/// Every container whose STATED length is admitted, generated with ffmpeg's
/// own encoders: the measurement that admits them, repeated wherever the
/// tests run.
///
/// The oracle is the tone's CONTENT length. A stated length may exceed the
/// content by a few milliseconds (codec priming the container counts) and must
/// never fall short of it, the direction that discards speech. The decode must
/// not run past the stated length either, or the bound would cut off audio an
/// engine receives. That second property is the pinned release's: ffmpeg 6.1
/// ignores an ISO media edit list's duration and decodes a 20.3 s AAC tone
/// 17.5 ms long (its untrimmed end padding), which is how these tests failed
/// on an unpinned CI runner.
#[tokio::test]
async fn stated_lengths_agree_with_a_full_decode() {
    let dir = tempfile::tempdir().expect("tempdir");
    for (name, codec, container) in [
        (
            "pcm.wav",
            &["-acodec", "pcm_s16le"][..],
            StatingContainer::PcmWav,
        ),
        (
            "lossless.flac",
            &["-acodec", "flac"][..],
            StatingContainer::Flac,
        ),
        (
            "aac.m4a",
            &["-acodec", "aac"][..],
            StatingContainer::IsoMedia,
        ),
        ("flac.ogg", &["-acodec", "flac"][..], StatingContainer::Ogg),
        (
            "aac.mkv",
            &["-acodec", "aac"][..],
            StatingContainer::Matroska,
        ),
        ("wma.wma", &["-acodec", "wmav2"][..], StatingContainer::Asf),
    ] {
        let source = dir.path().join(name);
        encode(&source, ToneMs::LONG, codec);
        let probed = MediaProbe::new(&source).duration().await.expect("probes");
        assert_eq!(probed.basis(), DurationBasis::Stated(container), "{name}");
        let (stated, content) = (probed.length().0, ToneMs::LONG.0);
        assert!(
            stated >= content && stated - content <= 50,
            "{name}: stated {stated} ms for {content} ms of content"
        );
        let decoded = decoded_ms(&source, 44_100);
        assert!(
            decoded <= stated,
            "{name}: decoded {decoded} ms, past the stated {stated} ms"
        );
    }
}

/// A FLAC written to a pipe cannot seek back to state its sample count. Its
/// packets sum exactly, so it is walked rather than refused.
#[tokio::test]
async fn a_flac_that_states_no_length_is_walked() {
    let dir = tempfile::tempdir().expect("tempdir");
    let source = dir.path().join("streamed.flac");
    let output = MediaTool::Ffmpeg
        .command()
        .args([
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            &ToneMs::LONG.source(),
            "-acodec",
            "flac",
            "-f",
            "flac",
            "-",
        ])
        .output()
        .expect("test requires ffmpeg");
    assert!(output.status.success());
    std::fs::write(&source, output.stdout).expect("write");

    let probed = MediaProbe::new(&source).duration().await.expect("probes");
    assert_eq!(
        probed.basis(),
        DurationBasis::Walked(WalkReason::LengthNotStated)
    );
    assert_eq!(probed.length(), DurationMs(decoded_ms(&source, 44_100)));
}

/// A container nobody has measured is refused by its demuxer's name rather
/// than trusted: AIFF is a real format ffmpeg reads, and admitting it is a
/// measured decision this crate has not made.
#[tokio::test]
async fn an_unmeasured_container_is_refused_by_name() {
    let dir = tempfile::tempdir().expect("tempdir");
    let source = dir.path().join("tone.aiff");
    encode(&source, ToneMs(2_000), &["-acodec", "pcm_s16be"]);
    match MediaProbe::new(&source).duration().await {
        Err(ProbeError::UnmeasuredContainer { demuxer, .. }) => assert_eq!(demuxer, "aiff"),
        other => panic!("expected a refusal, got {other:?}"),
    }
}

/// A file ffprobe cannot read is refused with ffprobe's own words.
#[tokio::test]
async fn an_unreadable_file_is_refused_with_ffprobe_diagnostics() {
    let dir = tempfile::tempdir().expect("tempdir");
    let source = dir.path().join("noise.mp3");
    std::fs::write(&source, b"this is not audio").expect("write");
    match MediaProbe::new(&source).duration().await {
        Err(ProbeError::Refused { diagnostics, .. }) => {
            assert!(!diagnostics.is_empty(), "ffprobe's reason is kept")
        }
        other => panic!("expected a refusal, got {other:?}"),
    }
}

/// An MP4 whose video runs past its audio: ffprobe states 10 s for the file
/// and 6 s for the audio stream. The probe measures AUDIO, so it takes the
/// stream's length: the tone's, not the video's.
#[tokio::test]
async fn an_mp4_with_longer_video_is_measured_by_its_audio() {
    let dir = tempfile::tempdir().expect("tempdir");
    let source = dir.path().join("lecture.mp4");
    let tone = ToneMs(6_000);
    let status = MediaTool::Ffmpeg
        .command()
        .args([
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=10:size=64x64:rate=5",
            "-f",
            "lavfi",
            "-i",
            &tone.source(),
            "-map",
            "0:v",
            "-map",
            "1:a",
            "-c:v",
            "mpeg4",
            "-acodec",
            "aac",
        ])
        .arg(&source)
        .status()
        .expect("test requires ffmpeg");
    assert!(status.success());
    assert!(
        container_duration_ms(&source) > 9_000.0,
        "premise: the file states the video's length"
    );
    let probed = MediaProbe::new(&source).duration().await.expect("probes");
    assert_eq!(probed.length(), DurationMs(tone.0));
    assert_eq!(
        probed.basis(),
        DurationBasis::Stated(StatingContainer::IsoMedia)
    );
    assert_eq!(decoded_ms(&source, 44_100), tone.0);
}
