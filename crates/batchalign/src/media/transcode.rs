//! Producing model-ready audio from arbitrary media.
//!
//! # Why this type exists
//!
//! Four call sites used to spell out the same ffmpeg invocation by hand:
//!
//! ```text
//! -y -nostdin -v error [-ss START -to END] -i SOURCE [-f f32le] -acodec pcm_{s16le,f32le} \
//!    -ar 16000 -ac 1 DESTINATION
//! ```
//!
//! differing only in whether a window was present and which PCM encoding came
//! out. Each then repeated the same three steps afterwards: check the exit
//! status, delete the partial output on failure, and build its own module's
//! "ffmpeg failed" error carrying stderr. And each classified a failure to
//! SPAWN separately from a failure to RUN.
//!
//! Naming the tool did not fix that, which is the lesson worth keeping. An
//! earlier pass gave `ffmpeg` one owner for its NAME, its spawn construction
//! and its availability predicate, and every one of those duplications
//! survived, because they do not live in the tool. They live in the OPERATION,
//! and the operation had no type. A well-typed noun with an untyped verb is a
//! defect factory: the verb is where the invariants are, so the verb is where
//! callers duplicate.
//!
//! # What this makes unrepresentable
//!
//! - An argv this crate does not support. There is no way to spell one.
//! - A window whose end does not follow its start. [`MediaWindow`] refuses it
//!   at construction, so the check cannot be forgotten by a new call site and
//!   cannot be reported (as it once was) as an I/O error.
//! - A partial output file surviving a failed transcode. The cleanup is inside
//!   [`Transcode::produce`], not restated by each caller.
//! - A spawn failure read as a transcode failure, or the reverse.
//!
//! # What it deliberately leaves to callers
//!
//! Whether a SUCCESSFUL transcode that produced zero bytes is an error.
//! `ffmpeg` exits 0 and writes an empty or truncated output rather than
//! failing, and the two consumers legitimately differ: one reports an
//! empty-segment error naming the window, the other has no window to name.
//! [`ProducedMedia`] therefore reports the byte length as a FACT and judges
//! nothing.
//!
//! This said `ffmpeg` exits 0 "when a requested window falls entirely past the
//! end of the source" until 2026-09-07. That is the unwitnessable cause
//! [`super::window::EmptyReason`] exists to replace, and it was already gone
//! from `error.rs`: a truncated write produces the same zero bytes from a
//! window well inside the file, and nothing on this path holds a source
//! duration to tell the two apart.

use std::ffi::OsString;
use std::path::{Path, PathBuf};

use super::probe::{MediaProbe, ProbeError};
use super::tools::{MediaTool, MediaToolError};
use super::window::DecodedFrames;
pub use super::window::{EmptyWindow, MediaWindow};
use crate::api::DurationMs;

/// How far the decoded audio may fall short of the source's declared length
/// when ffmpeg reported decoding errors, and still be admitted.
///
/// A damaged packet that ffmpeg CONCEALS keeps the timeline: the decode is as
/// long as the source. One it DROPS shortens the decode, and every later
/// sample, so every later word timing, moves earlier by the loss. The
/// tolerance absorbs codec priming and padding (tens of milliseconds for AAC)
/// and nothing that could be a lost packet run; a real loss is refused.
/// Measured on the recordings that motivated this (2026-09-24): seven MP4s
/// with AAC "channel element is not allocated" errors, every one decoding to
/// its container length to the millisecond.
pub const DAMAGE_SHORTFALL_TOLERANCE_MS: u64 = 100;

/// Whether a transcode decoded cleanly, or reported damage it concealed.
///
/// Carried on [`ProducedMedia`] so a caller that records provenance can say
/// which it was.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DecodeIntegrity {
    /// ffmpeg reported nothing.
    Clean,
    /// ffmpeg reported decoding errors, and the decode lost no audio.
    ConcealedDamage(ConcealedDamage),
}

/// A decode that reported errors yet runs as long as its source declares,
/// within [`DAMAGE_SHORTFALL_TOLERANCE_MS`]: the damage was concealed, not
/// dropped.
///
/// The fields are private and the only constructor is [`Self::admit`], which
/// applies the tolerance, so a value is the proof that the admission rule ran;
/// no caller can assemble one whose lengths disagree.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConcealedDamage {
    diagnostics: String,
    expected: DurationMs,
    decoded: DurationMs,
}

/// What a damaged decode lost: the refusal [`ConcealedDamage::admit`] returns,
/// handing ffmpeg's diagnostics back for the error that reports it.
#[derive(Clone, Debug, PartialEq, Eq)]
struct LostAudio {
    diagnostics: String,
    expected: DurationMs,
    decoded: DurationMs,
}

impl ConcealedDamage {
    /// The one statement of the admission rule. A decode LONGER than the
    /// source declares is not a loss (containers under-declare); only a
    /// shortfall beyond the tolerance is.
    fn admit(
        diagnostics: String,
        expected: DurationMs,
        decoded: DurationMs,
    ) -> Result<Self, LostAudio> {
        match expected.0.saturating_sub(decoded.0) <= DAMAGE_SHORTFALL_TOLERANCE_MS {
            true => Ok(Self {
                diagnostics,
                expected,
                decoded,
            }),
            false => Err(LostAudio {
                diagnostics,
                expected,
                decoded,
            }),
        }
    }

    /// ffmpeg's diagnostics, verbatim.
    pub fn diagnostics(&self) -> &str {
        &self.diagnostics
    }

    /// The source's declared length for the requested span.
    pub const fn expected(&self) -> DurationMs {
        self.expected
    }

    /// What was decoded.
    pub const fn decoded(&self) -> DurationMs {
        self.decoded
    }
}

/// The sample rate every ML model in this crate consumes.
///
/// Stated once. It was written at four call sites, which is four places to
/// disagree the day a model wants something else.
const MODEL_SAMPLE_RATE_HZ: u32 = 16_000;

/// Mono. Same reasoning as [`MODEL_SAMPLE_RATE_HZ`].
const MODEL_CHANNELS: u16 = 1;

/// How the produced PCM is encoded.
///
/// The codec and the container travel together because they are one decision:
/// raw float PCM needs `-f` to say so, while the WAV case infers its container
/// from the destination's extension.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PcmEncoding {
    /// 16-bit signed PCM in a WAV container, for the media conversion cache.
    S16LeWav,
    /// Raw 32-bit float PCM with no container, for worker-protocol artifacts.
    F32LeRaw,
}

impl PcmEncoding {
    const fn codec(self) -> &'static str {
        match self {
            Self::S16LeWav => "pcm_s16le",
            Self::F32LeRaw => "pcm_f32le",
        }
    }

    /// The explicit output format, when the destination's extension cannot say.
    const fn container_format(self) -> Option<&'static str> {
        match self {
            Self::S16LeWav => None,
            Self::F32LeRaw => Some("f32le"),
        }
    }
}

/// A transcode this crate knows how to ask for.
#[derive(Clone, Debug)]
pub struct Transcode {
    source: PathBuf,
    window: Option<MediaWindow>,
    encoding: PcmEncoding,
}

/// What a transcode produced, as facts its caller cannot otherwise get cheaply.
///
/// `byte_len` is here because the caller that cares would otherwise stat the
/// file again, and because a zero-length output is the one failure ffmpeg
/// signals with a SUCCESSFUL exit.
///
/// It deliberately does NOT carry the destination path: the caller passed that
/// in and still holds it, so returning a clone of it was an allocation handed
/// back to whoever supplied it. `#[must_use]` is likewise absent, because
/// `produce` returns `Result`, which already carries it, and the marker was
/// forcing `let _produced = produced?;` at the two sites that need only the
/// error.
#[derive(Debug, PartialEq, Eq)]
pub struct ProducedMedia {
    /// How many bytes it holds.
    byte_len: u64,
    /// The sample rate the audio was produced at.
    ///
    /// Reported rather than left for the caller to restate. `artifacts_v2`
    /// declared `SampleRateHzV2(16_000)` into the descriptor Python consumes,
    /// six lines after asking for audio at whatever this constant says, so
    /// changing the constant left the descriptor asserting the old value.
    sample_rate_hz: u32,
    /// How many channels it has, for the same reason. `u16` because that is
    /// the width the worker protocol's own channel count uses; a fact that
    /// travels should not change type on the way.
    channels: u16,
    /// Whether the decode was clean or concealed reported damage.
    integrity: DecodeIntegrity,
}

impl ProducedMedia {
    /// Byte length observed after strict conversion.
    pub const fn byte_len(&self) -> u64 {
        self.byte_len
    }

    /// Sample rate requested by the checked conversion.
    pub const fn sample_rate_hz(&self) -> u32 {
        self.sample_rate_hz
    }

    /// Channel count requested by the checked conversion.
    pub const fn channels(&self) -> u16 {
        self.channels
    }

    /// Whether the decode was clean or concealed reported damage.
    pub const fn integrity(&self) -> &DecodeIntegrity {
        &self.integrity
    }
}

/// Why a transcode did not produce its file.
///
/// EVERY variant carries the input it was reading. That is what lets each
/// consuming module write a total `From` impl and use `?`, instead of a
/// hand-written mapping function that has to be passed the path separately.
/// Two such functions existed, one per module, and were the last duplicated
/// reading of this failure.
///
/// The path field is `input`, not `source`: `thiserror` reads a field named
/// `source` as the underlying error, and this one is a path.
#[derive(Debug, thiserror::Error)]
pub enum TranscodeError {
    /// `ffmpeg` is not installed, or not on `PATH`.
    #[error(
        "ffmpeg not found in PATH. Cannot transcode {input}.\n\
         Hint: install ffmpeg (https://ffmpeg.org/download.html) \
         or convert your input audio to .wav beforehand."
    )]
    FfmpegMissing {
        /// The media it was asked to read.
        input: String,
    },
    /// `ffmpeg` ran and refused the work.
    #[error("ffmpeg failed to transcode {input}: {stderr}")]
    Failed {
        /// The media it was asked to read.
        input: String,
        /// What ffmpeg said, which is the operator-facing detail.
        stderr: String,
    },
    /// `ffmpeg` exists but could not be started.
    #[error("could not run ffmpeg on {input}: {source}")]
    Spawn {
        /// The media it was asked to read.
        input: String,
        /// What the operating system said.
        source: std::io::Error,
    },
    /// ffmpeg reported decoding errors AND the decode lost audio: the output
    /// is shorter than the source declares by more than
    /// [`DAMAGE_SHORTFALL_TOLERANCE_MS`], so every later timing would shift.
    #[error(
        "ffmpeg reported decoding errors on {input} and the decode lost audio: \
         {decoded_ms} ms decoded of {expected_ms} ms expected (tolerance \
         {DAMAGE_SHORTFALL_TOLERANCE_MS} ms): {stderr}"
    )]
    DamagedAudioLost {
        /// The media it was asked to read.
        input: String,
        /// The source's declared length for the requested span, in ms.
        expected_ms: u64,
        /// What was decoded, in ms.
        decoded_ms: u64,
        /// What ffmpeg said.
        stderr: String,
    },
    /// ffmpeg reported decoding errors and the lengths needed to admit the
    /// decode could not be measured, so it is refused rather than trusted.
    #[error(
        "ffmpeg reported decoding errors on {input} and the decode could not be measured: {source}"
    )]
    DamageUnmeasured {
        /// The media it was asked to read.
        input: String,
        /// Why the length could not be read.
        source: ProbeError,
    },
    /// The transcode succeeded but its output could not be inspected.
    #[error("could not inspect output transcoded from {input}: {source}")]
    Inspect {
        /// The media it was asked to read.
        input: String,
        /// What the operating system said.
        source: std::io::Error,
    },
}

impl Transcode {
    /// Transcode a whole file.
    pub fn whole(source: impl Into<PathBuf>, encoding: PcmEncoding) -> Self {
        Self {
            source: source.into(),
            window: None,
            encoding,
        }
    }

    /// Transcode one window of a file.
    pub fn window(source: impl Into<PathBuf>, window: MediaWindow, encoding: PcmEncoding) -> Self {
        Self {
            source: source.into(),
            window: Some(window),
            encoding,
        }
    }

    /// Write `destination`, removing a partial file if ffmpeg refuses.
    ///
    /// Blocking, and deliberately: every caller already runs on a dedicated
    /// thread because a transcode is long work, and two of them hold a file
    /// lock across it.
    pub fn produce(&self, destination: &Path) -> Result<ProducedMedia, TranscodeError> {
        let input = self.source.display().to_string();
        let output =
            MediaTool::Ffmpeg
                .run(self.args(destination))
                .map_err(|error| match error {
                    MediaToolError::NotInstalled(_) => TranscodeError::FfmpegMissing {
                        input: input.clone(),
                    },
                    MediaToolError::Spawn { source, .. } => TranscodeError::Spawn {
                        input: input.clone(),
                        source,
                    },
                })?;

        if !output.status.success() {
            discard_partial(destination);
            return Err(TranscodeError::Failed {
                input,
                stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            });
        }

        let byte_len = match std::fs::metadata(destination) {
            Ok(metadata) => metadata.len(),
            Err(source) => return Err(TranscodeError::Inspect { input, source }),
        };

        // `-v error` makes any decoder diagnostic visible on stderr. A clean
        // decode says nothing; one that said something is admitted only when
        // it lost no audio, and then with a warning.
        let integrity = match output.stderr.is_empty() {
            true => DecodeIntegrity::Clean,
            false => {
                let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
                match self.admit_damaged(destination, byte_len, &input, stderr) {
                    Ok(concealed) => DecodeIntegrity::ConcealedDamage(concealed),
                    Err(error) => {
                        discard_partial(destination);
                        return Err(error);
                    }
                }
            }
        };

        Ok(ProducedMedia {
            byte_len,
            sample_rate_hz: MODEL_SAMPLE_RATE_HZ,
            channels: MODEL_CHANNELS,
            integrity,
        })
    }

    /// Admit a decode that reported errors, or refuse it: measure the
    /// source's declared length for the requested span against what was
    /// decoded, and let [`ConcealedDamage::admit`] rule. Asked only after
    /// ffmpeg reported errors, so a clean decode pays for no probe.
    fn admit_damaged(
        &self,
        destination: &Path,
        byte_len: u64,
        input: &str,
        stderr: String,
    ) -> Result<ConcealedDamage, TranscodeError> {
        let unmeasured = |source| TranscodeError::DamageUnmeasured {
            input: input.to_owned(),
            source,
        };
        let source = MediaProbe::new(&self.source)
            .duration_blocking()
            .map_err(unmeasured)?;
        let expected = DurationMs(match self.window {
            // A window reaching past the source's end can only decode what the
            // source holds, so the expectation is clipped to it.
            Some(window) => source
                .0
                .min(window.end().get())
                .saturating_sub(window.start().get()),
            None => source.0,
        });
        let decoded = match self.encoding {
            // Raw float PCM: the length follows from the frame count.
            PcmEncoding::F32LeRaw => DurationMs(match DecodedFrames::measure_f32le(byte_len) {
                DecodedFrames::Frames(frames) => {
                    frames.get() * 1000 / u64::from(MODEL_SAMPLE_RATE_HZ)
                }
                DecodedFrames::None(_) => 0,
            }),
            // A WAV's header size varies, so its length is read, not computed.
            PcmEncoding::S16LeWav => MediaProbe::new(destination)
                .duration_blocking()
                .map_err(unmeasured)?,
        };
        match ConcealedDamage::admit(stderr, expected, decoded) {
            Ok(concealed) => {
                tracing::warn!(
                    input,
                    expected_ms = expected.0,
                    decoded_ms = decoded.0,
                    diagnostics = %concealed.diagnostics().trim(),
                    "decoded with errors that lost no audio; admitted"
                );
                Ok(concealed)
            }
            Err(lost) => Err(TranscodeError::DamagedAudioLost {
                input: input.to_owned(),
                expected_ms: lost.expected.0,
                decoded_ms: lost.decoded.0,
                stderr: lost.diagnostics,
            }),
        }
    }

    /// The full argv, which exists in exactly this one place.
    fn args(&self, destination: &Path) -> Vec<OsString> {
        let mut args: Vec<OsString> = ["-y", "-nostdin", "-v", "error"]
            .into_iter()
            .map(OsString::from)
            .collect();
        if let Some(window) = self.window {
            args.extend(window.as_seek_args());
        }
        args.push(OsString::from("-i"));
        args.push(self.source.clone().into_os_string());
        if let Some(format) = self.encoding.container_format() {
            args.push(OsString::from("-f"));
            args.push(OsString::from(format));
        }
        args.extend([
            OsString::from("-acodec"),
            OsString::from(self.encoding.codec()),
            OsString::from("-ar"),
            OsString::from(MODEL_SAMPLE_RATE_HZ.to_string()),
            OsString::from("-ac"),
            OsString::from(MODEL_CHANNELS.to_string()),
            destination.to_path_buf().into_os_string(),
        ]);
        args
    }
}

/// Remove a partial output after a refusal: the one statement of this cleanup
/// for every refusal path in [`Transcode::produce`].
fn discard_partial(destination: &Path) {
    let _ = std::fs::remove_file(destination);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time::FileMs;

    /// A truncated PCM sample is a real decoder error that ffmpeg tolerates
    /// with exit zero and all but one sample still emitted: the damage lost
    /// no audio, so the decode is admitted, its diagnostics kept.
    #[test]
    fn decoder_errors_that_lose_no_audio_are_admitted_with_their_diagnostics() {
        assert!(MediaTool::Ffmpeg.banner().is_some(), "test requires ffmpeg");
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("truncated.wav");
        let destination = dir.path().join("output.wav");
        let mut wav = Vec::new();
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&20_036u32.to_le_bytes());
        wav.extend_from_slice(b"WAVEfmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
        wav.extend_from_slice(&1u16.to_le_bytes()); // mono
        wav.extend_from_slice(&16_000u32.to_le_bytes());
        wav.extend_from_slice(&32_000u32.to_le_bytes());
        wav.extend_from_slice(&2u16.to_le_bytes());
        wav.extend_from_slice(&16u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&20_000u32.to_le_bytes());
        wav.resize(44 + 19_999, 0); // final sample is missing one byte
        std::fs::write(&source, wav).unwrap();

        let permissive = MediaTool::Ffmpeg
            .command()
            .args(["-nostdin", "-v", "error", "-i"])
            .arg(&source)
            .args(["-f", "s16le", "-"])
            .output()
            .unwrap();
        assert!(permissive.status.success());
        assert!(!permissive.stdout.is_empty());
        assert!(!permissive.stderr.is_empty());

        let produced = Transcode::whole(&source, PcmEncoding::S16LeWav)
            .produce(&destination)
            .expect("damage that lost no audio is admitted");
        match produced.integrity() {
            DecodeIntegrity::ConcealedDamage(concealed) => {
                assert!(
                    !concealed.diagnostics().is_empty(),
                    "ffmpeg's report is kept"
                );
                assert!(
                    concealed.expected().0.saturating_sub(concealed.decoded().0)
                        <= DAMAGE_SHORTFALL_TOLERANCE_MS
                );
            }
            DecodeIntegrity::Clean => panic!("the decoder reported an error; it was not clean"),
        }
        assert!(destination.exists());
    }

    /// The admission rule itself: a shortfall up to the tolerance is concealed
    /// damage, one past it is lost audio, and a decode longer than the
    /// container declares is never a loss.
    #[test]
    fn damage_is_admitted_only_when_no_audio_was_lost() {
        let admit = |expected: u64, decoded: u64| {
            ConcealedDamage::admit(
                "damaged packet".into(),
                DurationMs(expected),
                DurationMs(decoded),
            )
            .is_ok()
        };
        assert!(admit(785_110, 785_110));
        assert!(admit(785_110, 785_110 - DAMAGE_SHORTFALL_TOLERANCE_MS));
        assert!(!admit(785_110, 785_110 - DAMAGE_SHORTFALL_TOLERANCE_MS - 1));
        assert!(
            admit(1_000, 1_050),
            "a decode longer than declared is not a loss"
        );
        let lost = ConcealedDamage::admit(
            "damaged packet".into(),
            DurationMs(2_000),
            DurationMs(1_000),
        )
        .expect_err("a second of audio was lost");
        assert_eq!(
            lost.diagnostics, "damaged packet",
            "the refusal keeps ffmpeg's report"
        );
    }

    fn ms(value: u64) -> FileMs {
        FileMs::new(value)
    }

    /// An empty window cannot be built, so no call site can forget to check.
    ///
    /// POLICY: a zero-length window is refused rather than transcoded to
    /// nothing, because every consumer treats empty audio as a failure and one
    /// of them used to report this as `io::ErrorKind::InvalidInput`, i.e. as a
    /// filesystem problem.
    #[test]
    fn a_window_that_holds_nothing_cannot_be_constructed() {
        assert!(MediaWindow::new(ms(500), ms(500)).is_err());
        assert!(MediaWindow::new(ms(500), ms(499)).is_err());
        assert!(MediaWindow::new(ms(500), ms(501)).is_ok());
    }

    /// The argv, pinned. This is the closest thing to a wire format the crate
    /// has: it is what reaches `execvp`, and no type can describe what ffmpeg
    /// will accept.
    #[test]
    fn a_whole_file_wav_transcode_reports_decoder_errors_without_stopping() {
        let args = Transcode::whole("/in.mp4", PcmEncoding::S16LeWav).args(Path::new("/out.wav"));
        let rendered: Vec<_> = args.iter().map(|a| a.to_string_lossy()).collect();
        assert_eq!(
            rendered,
            [
                "-y",
                "-nostdin",
                "-v",
                "error",
                "-i",
                "/in.mp4",
                "-acodec",
                "pcm_s16le",
                "-ar",
                "16000",
                "-ac",
                "1",
                "/out.wav",
            ]
        );
    }

    /// The windowed raw-PCM form: seek args before `-i`, and an explicit `-f`
    /// because raw PCM has no container for the extension to imply.
    #[test]
    fn a_windowed_raw_transcode_seeks_before_input_and_states_its_format() {
        let window = MediaWindow::new(ms(1_500), ms(2_250)).expect("non-empty window");
        let args =
            Transcode::window("/in.wav", window, PcmEncoding::F32LeRaw).args(Path::new("/out.pcm"));
        let rendered: Vec<_> = args.iter().map(|a| a.to_string_lossy()).collect();
        assert_eq!(
            rendered,
            [
                "-y",
                "-nostdin",
                "-v",
                "error",
                "-ss",
                "1.500",
                "-to",
                "2.250",
                "-i",
                "/in.wav",
                "-f",
                "f32le",
                "-acodec",
                "pcm_f32le",
                "-ar",
                "16000",
                "-ac",
                "1",
                "/out.pcm",
            ]
        );
    }
}
