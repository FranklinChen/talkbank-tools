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

use std::path::PathBuf;

use super::tools::{MediaTool, MediaToolError};
use crate::api::DurationMs;

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
    #[error("ffprobe could not read {input}")]
    Refused {
        /// The media it was asked to read.
        input: String,
    },
    /// `ffprobe` answered with something this crate cannot read as a duration.
    #[error("ffprobe reported an unreadable duration for {input}: {answer:?}")]
    Unreadable {
        /// The media it was asked to read.
        input: String,
        /// What it printed, so an operator can see it.
        answer: String,
    },
    /// `ffprobe` exists but could not be started.
    #[error("could not run ffprobe on {input}: {source}")]
    Spawn {
        /// The media it was asked to read.
        input: String,
        /// What the operating system said.
        source: std::io::Error,
    },
}

impl MediaProbe {
    /// Probe `source`.
    pub fn new(source: impl Into<PathBuf>) -> Self {
        Self {
            source: source.into(),
        }
    }

    /// How long the audio runs.
    pub async fn duration(&self) -> Result<DurationMs, ProbeError> {
        let output = MediaTool::Ffprobe.run_async(self.duration_args()).await;
        self.read_duration(output)
    }

    /// How long the audio runs, for a caller already on a blocking thread
    /// (a transcode checking what a damaged decode produced). The same
    /// question and the same reading of the answer as [`Self::duration`];
    /// only the spawn differs.
    pub fn duration_blocking(&self) -> Result<DurationMs, ProbeError> {
        let output = MediaTool::Ffprobe.run(self.duration_args());
        self.read_duration(output)
    }

    /// The one statement of the duration question.
    fn duration_args(&self) -> [&std::ffi::OsStr; 7] {
        [
            "-v".as_ref(),
            "quiet".as_ref(),
            "-show_entries".as_ref(),
            "format=duration".as_ref(),
            "-of".as_ref(),
            "default=noprint_wrappers=1:nokey=1".as_ref(),
            self.source.as_os_str(),
        ]
    }

    /// The one reading of ffprobe's answer, for both spawns.
    fn read_duration(
        &self,
        output: Result<std::process::Output, MediaToolError>,
    ) -> Result<DurationMs, ProbeError> {
        let input = self.source.display().to_string();
        let output = output.map_err(|error| match error {
            MediaToolError::NotInstalled(_) => ProbeError::FfprobeMissing {
                input: input.clone(),
            },
            MediaToolError::Spawn { source, .. } => ProbeError::Spawn {
                input: input.clone(),
                source,
            },
        })?;

        if !output.status.success() {
            return Err(ProbeError::Refused { input });
        }

        let answer = String::from_utf8_lossy(&output.stdout).trim().to_owned();
        let seconds: f64 = answer.parse().map_err(|_| ProbeError::Unreadable {
            input,
            answer: answer.clone(),
        })?;
        Ok(DurationMs((seconds * 1000.0) as u64))
    }
}
