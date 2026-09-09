//! Producing model-ready audio from arbitrary media.
//!
//! # Why this type exists
//!
//! Four call sites used to spell out the same ffmpeg invocation by hand:
//!
//! ```text
//! -y [-ss START -to END] -i SOURCE [-f f32le] -acodec pcm_{s16le,f32le} \
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

use super::tools::{MediaTool, MediaToolError};
pub use super::window::{EmptyWindow, MediaWindow};

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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProducedMedia {
    /// How many bytes it holds.
    pub byte_len: u64,
    /// The sample rate the audio was produced at.
    ///
    /// Reported rather than left for the caller to restate. `artifacts_v2`
    /// declared `SampleRateHzV2(16_000)` into the descriptor Python consumes,
    /// six lines after asking for audio at whatever this constant says, so
    /// changing the constant left the descriptor asserting the old value.
    pub sample_rate_hz: u32,
    /// How many channels it has, for the same reason. `u16` because that is
    /// the width the worker protocol's own channel count uses; a fact that
    /// travels should not change type on the way.
    pub channels: u16,
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
            // The one statement of this cleanup. Four call sites each had their
            // own, and a fifth would have had to remember.
            let _ = std::fs::remove_file(destination);
            return Err(TranscodeError::Failed {
                input,
                stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            });
        }

        let byte_len = std::fs::metadata(destination)
            .map_err(|source| TranscodeError::Inspect { input, source })?
            .len();
        Ok(ProducedMedia {
            byte_len,
            sample_rate_hz: MODEL_SAMPLE_RATE_HZ,
            channels: MODEL_CHANNELS,
        })
    }

    /// The full argv, which exists in exactly this one place.
    fn args(&self, destination: &Path) -> Vec<OsString> {
        let mut args = vec![OsString::from("-y")];
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time::FileMs;

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
    fn a_whole_file_wav_transcode_spells_the_argv_callers_used_to_write() {
        let args = Transcode::whole("/in.mp4", PcmEncoding::S16LeWav).args(Path::new("/out.wav"));
        let rendered: Vec<_> = args.iter().map(|a| a.to_string_lossy()).collect();
        assert_eq!(
            rendered,
            [
                "-y",
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
