//! Preservation-oriented audio export, distinct from model-facing transcoding.
//!
//! The plan owns both paths. Encoding produces a private temporary artifact;
//! only inspected output can consume the new-only publication operation.

use std::ffi::OsString;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use tempfile::NamedTempFile;

use super::probe::{AudioDuration, MediaProbe, ObservedSingleAudio, ProbeError};
use super::tools::{MediaTool, MediaToolError};

/// User-selected output encoding; neither variant is model audio.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
pub enum AudioExportFormat {
    /// PCM16 WAVE, retaining the input sample rate and channel count.
    Wav,
    /// VBR-quality-2 MP3, retaining an MP3-compatible sample rate and channel count.
    Mp3,
}

impl AudioExportFormat {
    /// Canonical output suffix and container name.
    #[must_use]
    pub const fn extension(self) -> &'static str {
        match self {
            Self::Wav => "wav",
            Self::Mp3 => "mp3",
        }
    }

    const fn codec(self) -> &'static str {
        match self {
            Self::Wav => "pcm_s16le",
            Self::Mp3 => "libmp3lame",
        }
    }

    const fn output_codec(self) -> &'static str {
        match self {
            Self::Wav => "pcm_s16le",
            Self::Mp3 => "mp3",
        }
    }
}

/// Why source admission, encoding, inspection or publication refused.
#[derive(Debug, thiserror::Error)]
pub enum AudioExportError {
    /// An export never replaces an existing file or symlink.
    #[error("audio export destination already exists: {0}")]
    ExistingDestination(PathBuf),
    /// A filesystem operation failed.
    #[error("audio export filesystem operation failed: {0}")]
    Io(#[from] std::io::Error),
    /// A required external tool could not be started.
    #[error("audio export could not start {tool}: {detail}")]
    Tool {
        /// The program that could not be started.
        tool: &'static str,
        /// Missing-program or operating-system detail.
        detail: String,
    },
    /// A tool refused the source or encoding; its diagnostic is retained.
    #[error("audio export {tool} refused: {diagnostics}")]
    Refused {
        /// The program that refused its input or encoding.
        tool: &'static str,
        /// Original producer diagnostics, not inferred CHAT invalidity.
        diagnostics: String,
    },
    /// A probe did not establish one usable audio stream.
    #[error("audio export stream admission refused: {0}")]
    Stream(String),
    /// Preservation needs an encoding that supports the source layout.
    #[error("MP3 cannot preserve {channels} channels at {sample_rate} Hz; export WAV instead")]
    Mp3Layout {
        /// Source channel count that preservation would require.
        channels: u32,
        /// Source sample rate that preservation would require.
        sample_rate: u32,
    },
    /// The duration owner refused empty or unmeasurable audio.
    #[error("{0}")]
    Duration(#[from] ProbeError),
    /// An encoder returned success but produced different audio properties.
    #[error("audio export producer changed the requested layout or encoding")]
    ProducerLayout,
}

impl From<MediaToolError> for AudioExportError {
    fn from(error: MediaToolError) -> Self {
        match error {
            MediaToolError::NotInstalled(tool) => Self::Tool {
                tool: tool.program(),
                detail: "not installed or not on PATH".into(),
            },
            MediaToolError::Spawn { tool, source } => Self::Tool {
                tool: tool.program(),
                detail: source.to_string(),
            },
        }
    }
}

async fn inspect_stream(source: &Path) -> Result<ObservedSingleAudio, AudioExportError> {
    MediaProbe::new(source)
        .inspect_single_audio()
        .await
        .map_err(|error| {
            let tool = MediaTool::Ffprobe.program();
            match error {
                ProbeError::FfprobeMissing { .. } => AudioExportError::Tool {
                    tool,
                    detail: "not installed or not on PATH".to_owned(),
                },
                ProbeError::Spawn { source, .. } => AudioExportError::Tool {
                    tool,
                    detail: source.to_string(),
                },
                ProbeError::Refused { diagnostics, .. } => {
                    AudioExportError::Refused { tool, diagnostics }
                }
                ProbeError::Unreadable { detail, .. } => AudioExportError::Stream(detail),
                other => AudioExportError::Duration(other),
            }
        })
}

/// An absent destination with its admitted parent carried into encoding.
#[derive(Debug)]
struct NewDestination {
    path: PathBuf,
    parent: PathBuf,
}

impl NewDestination {
    fn admit(destination: &Path) -> Result<Self, AudioExportError> {
        let parent = destination
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."))
            .canonicalize()?;
        let name = destination
            .file_name()
            .ok_or_else(|| AudioExportError::Stream("destination has no filename".into()))?;
        let path = parent.join(name);
        match std::fs::symlink_metadata(&path) {
            Ok(_) => return Err(AudioExportError::ExistingDestination(path)),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
        Ok(Self { path, parent })
    }

    fn verify_parent(&self) -> Result<(), AudioExportError> {
        if self.parent.canonicalize()? != self.parent {
            return Err(AudioExportError::Stream(
                "export destination directory changed after admission".into(),
            ));
        }
        Ok(())
    }
}

/// A primary export may optionally bind one staged destination.
#[derive(Debug)]
pub struct PrimaryAudioExport(AudioExportPlan);

/// Source-bound encoding and new-only destination admission.
#[derive(Debug)]
pub struct AudioExportPlan {
    source: PathBuf,
    destination: NewDestination,
    format: AudioExportFormat,
    staged_destination: Option<NewDestination>,
}

impl PrimaryAudioExport {
    /// Admit a regular source and an absent destination in an existing directory.
    ///
    /// The final atomic no-clobber operation repeats destination exclusion:
    /// a file appearing after this check is never overwritten.
    pub fn admit(
        source: &Path,
        destination: &Path,
        format: AudioExportFormat,
    ) -> Result<Self, AudioExportError> {
        let source = source.canonicalize()?;
        if !source.metadata()?.is_file() {
            return Err(AudioExportError::Stream(
                "source is not a regular file".into(),
            ));
        }
        let destination = NewDestination::admit(destination)?;
        if destination.path == source {
            return Err(AudioExportError::ExistingDestination(destination.path));
        }
        Ok(Self(AudioExportPlan {
            source,
            destination,
            format,
            staged_destination: None,
        }))
    }

    /// Bind one independent managed download copy before encoding.
    pub fn with_staged_copy(self, destination: &Path) -> Result<AudioExportPlan, AudioExportError> {
        let destination = NewDestination::admit(destination)?;
        let mut plan = self.0;
        if destination.path == plan.source || destination.path == plan.destination.path {
            return Err(AudioExportError::Stream(
                "staged export destination is not independent".into(),
            ));
        }
        plan.staged_destination = Some(destination);
        Ok(plan)
    }

    /// Fix this plan's destination set to its primary path.
    #[must_use]
    pub fn primary_only(self) -> AudioExportPlan {
        self.0
    }

    /// Encode only the primary export.
    pub async fn encode(self) -> Result<VerifiedAudioExport, AudioExportError> {
        self.primary_only().encode().await
    }
}

impl AudioExportPlan {
    /// Encode in a private adjacent file; inspect it before producing publication proof.
    /// Cancellation drops the temporary artifact and kills the owned tool process.
    pub async fn encode(self) -> Result<VerifiedAudioExport, AudioExportError> {
        let input = inspect_stream(&self.source).await?;
        let sample_rate = input.layout().sample_rate.get();
        let channels = input.layout().channels.get();
        if self.format == AudioExportFormat::Mp3
            && (channels > 2
                || ![
                    8_000, 11_025, 12_000, 16_000, 22_050, 24_000, 32_000, 44_100, 48_000,
                ]
                .contains(&sample_rate))
        {
            return Err(AudioExportError::Mp3Layout {
                channels,
                sample_rate,
            });
        }
        let temporary = tempfile::Builder::new()
            .prefix(".audio-export-")
            .suffix(&format!(".{}", self.format.extension()))
            .tempfile_in(&self.destination.parent)?;
        let mut args: Vec<OsString> = ["-nostdin", "-v", "error", "-xerror", "-y", "-i"]
            .into_iter()
            .map(OsString::from)
            .collect();
        args.push(self.source.as_os_str().to_owned());
        args.extend(
            [
                "-map",
                "0:a:0",
                "-vn",
                "-sn",
                "-dn",
                "-c:a",
                self.format.codec(),
            ]
            .into_iter()
            .map(OsString::from),
        );
        args.extend([
            OsString::from("-ar"),
            sample_rate.to_string().into(),
            OsString::from("-ac"),
            channels.to_string().into(),
        ]);
        if self.format == AudioExportFormat::Mp3 {
            args.extend([OsString::from("-q:a"), OsString::from("2")]);
        }
        args.extend([
            OsString::from("-f"),
            self.format.extension().into(),
            temporary.path().as_os_str().to_owned(),
        ]);
        let output = MediaTool::Ffmpeg.run_async(args).await?;
        if !output.status.success() {
            return Err(AudioExportError::Refused {
                tool: MediaTool::Ffmpeg.program(),
                diagnostics: String::from_utf8_lossy(&output.stderr).into_owned(),
            });
        }
        let encoded = inspect_stream(temporary.path()).await?;
        if encoded.layout() != input.layout() || encoded.codec() != self.format.output_codec() {
            return Err(AudioExportError::ProducerLayout);
        }
        let duration = encoded.duration().await?;
        temporary.as_file().sync_all()?;
        let staged = if let Some(destination) = self.staged_destination {
            let staged = tempfile::Builder::new()
                .prefix(".audio-export-")
                .tempfile_in(&destination.parent)?;
            // Held handles prevent cancellation from recreating a dropped
            // temporary path in a delayed filesystem-copy task.
            let mut input = tokio::fs::File::from_std(temporary.reopen()?);
            let mut output = tokio::fs::File::from_std(staged.reopen()?);
            tokio::io::copy(&mut input, &mut output).await?;
            output.sync_all().await?;
            Some(StagedExport {
                temporary: staged,
                destination,
            })
        } else {
            None
        };
        Ok(VerifiedAudioExport {
            temporary,
            destination: self.destination,
            duration,
            staged,
        })
    }
}

/// Encoded and inspected artifact. No public constructor or replaceable target.
#[derive(Debug)]
pub struct VerifiedAudioExport {
    temporary: NamedTempFile,
    destination: NewDestination,
    duration: AudioDuration,
    staged: Option<StagedExport>,
}

#[derive(Debug)]
struct StagedExport {
    temporary: NamedTempFile,
    destination: NewDestination,
}

impl VerifiedAudioExport {
    /// Measured nonempty encoded duration; not a linguistic-quality claim.
    #[must_use]
    pub const fn duration(&self) -> AudioDuration {
        self.duration
    }

    /// Publish exactly the admitted destination, atomically refusing existing files.
    pub fn publish(self) -> Result<PathBuf, AudioExportError> {
        self.destination.verify_parent()?;
        if let Some(staged) = self.staged {
            staged.destination.verify_parent()?;
            staged
                .temporary
                .persist_noclobber(&staged.destination.path)
                .map_err(|error| AudioExportError::Io(error.error))?;
        }
        // Publish the user-visible file last. A final competing writer can
        // leave a private staged copy, but never a claimed successful result
        // or an overwritten file; two filesystems are not one transaction.
        self.temporary
            .persist_noclobber(&self.destination.path)
            .map_err(|error| AudioExportError::Io(error.error))?;
        Ok(self.destination.path)
    }
}

#[cfg(test)]
mod tests;
