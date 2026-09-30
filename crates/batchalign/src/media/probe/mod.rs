//! Reading properties of media without producing anything.
//!
//! The sibling of [`super::transcode`]: that one writes a file, this one
//! answers a question. Both exist for the same reason, which is that the
//! OPERATION is where the invariants live.
//!
//! Duration probing was the last production spawn outside these types, and it
//! escaped the earlier consolidation for a revealing reason: it is async, and
//! the spawn helper was blocking. So the split that kept it out was a LANGUAGE
//! boundary, not a domain one. Its cost was a total function that returned
//! `Option<u64>` and folded four different facts into `None`: ffprobe is not
//! installed, ffprobe was killed, ffprobe refused the file, and ffprobe printed
//! something unparseable. An operator seeing "no duration" could not tell a
//! missing dependency from a corrupt file.
//!
//! # Two questions, and why the second exists
//!
//! The duration is established in at most two ffprobe runs, and the order is
//! the type graph: [`header::HeaderAnswer`] (what container, does it state its
//! length) routes to either a stated length or a walk
//! ([`header::DurationRoute`]), and a walk's [`walk::WalkAnswer`] becomes the
//! length. Both ends are an [`AudioDuration`], which nothing else can build.
//!
//! Until 2026-09-30 there was one question, `format=duration`, trusted for
//! every file, which for some demuxers is a bitrate estimate; [`duration`]
//! tells that story and the type that closes it.

mod duration;
#[cfg(test)]
pub(crate) mod fixtures;
mod header;
#[cfg(test)]
mod tests;
mod walk;

use std::ffi::OsStr;
use std::path::PathBuf;

pub use duration::{AudioDuration, DurationBasis, StatingContainer, WalkReason};

use super::tools::{MediaTool, MediaToolError};

/// A property read from one media file.
#[derive(Clone, Debug)]
pub struct MediaProbe {
    source: PathBuf,
}

/// Why a probe could not answer.
///
/// Each variant is an operator action: install something, look at the file,
/// look at the machine. That is the whole point of not returning `None`.
#[derive(Debug, thiserror::Error)]
pub enum ProbeError {
    /// `ffprobe` is not installed, or not on `PATH`.
    #[error("ffprobe is not installed or not on PATH (probing {input})")]
    FfprobeMissing {
        /// The media it was asked to read.
        input: String,
    },
    /// `ffprobe` ran and refused the file.
    #[error("ffprobe could not read {input}: {diagnostics}")]
    Refused {
        /// The media it was asked to read.
        input: String,
        /// What ffprobe said, verbatim, so the reason reaches the operator.
        diagnostics: String,
    },
    /// `ffprobe` answered with something this crate cannot read.
    #[error("ffprobe's answer about {input} could not be read: {detail}")]
    Unreadable {
        /// The media it was asked to read.
        input: String,
        /// Which part of the answer, and why.
        detail: String,
    },
    /// `ffprobe` exists but could not be started.
    #[error("could not run ffprobe on {input}: {source}")]
    Spawn {
        /// The media it was asked to read.
        input: String,
        /// What the operating system said.
        source: std::io::Error,
    },
    /// The file holds no audio stream.
    #[error("{input} has no audio stream")]
    NoAudioStream {
        /// The media it was asked to read.
        input: String,
    },
    /// The file's container is not one whose duration has been measured
    /// against a decode, so neither its stated length nor a packet walk is
    /// known to be exact for it.
    #[error(
        "{input} is in a container ({demuxer}) whose duration batchalign has not verified; \
         convert it to WAV, FLAC, MP3 or M4A"
    )]
    UnmeasuredContainer {
        /// The media it was asked to read.
        input: String,
        /// ffprobe's name for the demuxer, verbatim.
        demuxer: String,
    },
    /// A container whose length is only trustworthy when stated, and which
    /// states none.
    #[error(
        "{input} does not state its length, and its container ({container:?}) cannot be measured by its packets"
    )]
    LengthNotStated {
        /// The media it was asked to read.
        input: String,
        /// The container that stated nothing.
        container: StatingContainer,
    },
    /// The audio stream's length is zero: the file holds no audio, so it
    /// cannot bound anything. Refused here, where the length is born, so no
    /// consumer receives an empty recording.
    #[error("{input} holds no audio (its audio stream has zero length)")]
    EmptyAudio {
        /// The media it was asked to read.
        input: String,
    },
    /// Some packets carried no duration, so their sum is only a lower bound.
    #[error("{count} packets of {input} carry no duration, so its length cannot be measured")]
    PacketsWithoutDuration {
        /// The media it was asked to read.
        input: String,
        /// How many.
        count: u64,
    },
}

/// The header question: the demuxer, the first audio stream's codec and
/// stated length, and the file's stated length. Reads headers only.
const HEADER_ENTRIES: &str = "format=format_name,duration:stream=codec_name,duration";

/// The walk question: every packet's duration and trims, and the clock they
/// are in. Demuxes the whole file without decoding it.
const WALK_ENTRIES: &str =
    "stream=time_base,sample_rate:packet=duration:packet_side_data=skip_samples,discard_padding";

impl MediaProbe {
    /// Probe `source`.
    pub fn new(source: impl Into<PathBuf>) -> Self {
        Self {
            source: source.into(),
        }
    }

    /// How long the audio runs, by the route that is exact for its container.
    pub async fn duration(&self) -> Result<AudioDuration, ProbeError> {
        let header = MediaTool::Ffprobe
            .run_async(self.args(HEADER_ENTRIES))
            .await;
        match self
            .answer::<header::HeaderAnswer>(header)?
            .route(&self.input())?
        {
            header::DurationRoute::Stated(duration) => Ok(duration),
            header::DurationRoute::Walk(reason) => {
                let walk = MediaTool::Ffprobe.run_async(self.args(WALK_ENTRIES)).await;
                self.answer::<walk::WalkAnswer>(walk)?
                    .length(&self.input(), reason)
            }
        }
    }

    /// How long the audio runs, for a caller already on a blocking thread
    /// (a transcode checking what a damaged decode produced). The same
    /// questions and the same reading of the answers as [`Self::duration`];
    /// only the spawns differ.
    pub fn duration_blocking(&self) -> Result<AudioDuration, ProbeError> {
        let header = MediaTool::Ffprobe.run(self.args(HEADER_ENTRIES));
        match self
            .answer::<header::HeaderAnswer>(header)?
            .route(&self.input())?
        {
            header::DurationRoute::Stated(duration) => Ok(duration),
            header::DurationRoute::Walk(reason) => {
                let walk = MediaTool::Ffprobe.run(self.args(WALK_ENTRIES));
                self.answer::<walk::WalkAnswer>(walk)?
                    .length(&self.input(), reason)
            }
        }
    }

    /// The argv for one question: the entries asked for, about the first
    /// audio stream, as compact JSON.
    fn args(&self, entries: &'static str) -> [&OsStr; 9] {
        [
            "-v".as_ref(),
            "error".as_ref(),
            "-select_streams".as_ref(),
            "a:0".as_ref(),
            "-show_entries".as_ref(),
            entries.as_ref(),
            "-of".as_ref(),
            "json=compact=1".as_ref(),
            self.source.as_os_str(),
        ]
    }

    /// The one reading of an ffprobe run, for both questions and both spawns:
    /// classify a failure to run, refuse a non-zero exit with ffprobe's own
    /// diagnostics, and deserialize the JSON it printed.
    fn answer<T: serde::de::DeserializeOwned>(
        &self,
        output: Result<std::process::Output, MediaToolError>,
    ) -> Result<T, ProbeError> {
        let input = self.input();
        let output = output.map_err(|error| match error {
            MediaToolError::NotInstalled(_) => ProbeError::FfprobeMissing {
                input: input.clone(),
            },
            MediaToolError::Spawn { source, .. } => ProbeError::Spawn {
                input: input.clone(),
                source,
            },
        })?;
        match output.status.success() {
            true => {
                serde_json::from_slice(&output.stdout).map_err(|error| ProbeError::Unreadable {
                    input,
                    detail: error.to_string(),
                })
            }
            false => Err(ProbeError::Refused {
                input,
                diagnostics: String::from_utf8_lossy(&output.stderr).trim().to_owned(),
            }),
        }
    }

    /// The source as reported in errors.
    fn input(&self) -> String {
        self.source.display().to_string()
    }
}
