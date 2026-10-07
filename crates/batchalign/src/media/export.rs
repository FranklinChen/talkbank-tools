//! Preservation-oriented audio export, distinct from model-facing transcoding.
//!
//! The plan owns both paths. Encoding produces a private temporary artifact;
//! only inspected output can consume the new-only publication operation.

use std::ffi::OsString;
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use tempfile::NamedTempFile;

use super::probe::{AudioDuration, MediaProbe, ProbeError};
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
    pub const fn extension(self) -> &'static str {
        match self {
            Self::Wav => "wav",
            Self::Mp3 => "mp3",
        }
    }

    fn codec(self) -> &'static str {
        match self {
            Self::Wav => "pcm_s16le",
            Self::Mp3 => "libmp3lame",
        }
    }

    fn output_codec(self) -> &'static str {
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

#[derive(Debug, PartialEq, Eq)]
struct AudioLayout {
    sample_rate: NonZeroU32,
    channels: NonZeroU32,
}

#[derive(Deserialize)]
struct StreamAnswer {
    streams: Vec<StreamWire>,
}

#[derive(Deserialize)]
struct StreamWire {
    sample_rate: String,
    channels: u32,
    codec_name: String,
}

struct InspectedStream {
    layout: AudioLayout,
    codec: String,
}

async fn inspect_stream(source: &Path) -> Result<InspectedStream, AudioExportError> {
    let args = [
        OsString::from("-v"),
        OsString::from("error"),
        OsString::from("-select_streams"),
        OsString::from("a"),
        OsString::from("-show_entries"),
        OsString::from("stream=sample_rate,channels,codec_name"),
        OsString::from("-of"),
        OsString::from("json"),
        source.as_os_str().to_owned(),
    ];
    let output = MediaTool::Ffprobe.run_async(args).await?;
    if !output.status.success() {
        return Err(AudioExportError::Refused {
            tool: MediaTool::Ffprobe.program(),
            diagnostics: String::from_utf8_lossy(&output.stderr).into_owned(),
        });
    }
    let answer: StreamAnswer = serde_json::from_slice(&output.stdout)
        .map_err(|error| AudioExportError::Stream(error.to_string()))?;
    let [stream]: [StreamWire; 1] =
        answer
            .streams
            .try_into()
            .map_err(|streams: Vec<StreamWire>| {
                AudioExportError::Stream(format!(
                    "expected one audio stream, observed {}",
                    streams.len()
                ))
            })?;
    let sample_rate = stream
        .sample_rate
        .parse::<NonZeroU32>()
        .map_err(|error| AudioExportError::Stream(error.to_string()))?;
    let channels = NonZeroU32::new(stream.channels)
        .ok_or_else(|| AudioExportError::Stream("zero audio channels".into()))?;
    Ok(InspectedStream {
        layout: AudioLayout {
            sample_rate,
            channels,
        },
        codec: stream.codec_name,
    })
}

/// Source-bound encoding and new-only destination admission.
#[derive(Debug)]
pub struct AudioExportPlan {
    source: PathBuf,
    destination: PathBuf,
    format: AudioExportFormat,
    staged_destination: Option<PathBuf>,
}

impl AudioExportPlan {
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
        require_absent(destination)?;
        let parent = destination
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        let name = destination
            .file_name()
            .ok_or_else(|| AudioExportError::Stream("destination has no filename".into()))?;
        let destination = parent.canonicalize()?.join(name);
        require_absent(&destination)?;
        Ok(Self {
            source,
            destination,
            format,
            staged_destination: None,
        })
    }

    /// Bind the managed download copy before encoding, not after obtaining proof.
    /// Both destinations must be absent and distinct from the source and each other.
    pub fn with_staged_copy(mut self, destination: &Path) -> Result<Self, AudioExportError> {
        require_absent(destination)?;
        let parent = destination.parent().ok_or_else(|| {
            AudioExportError::Stream("staged destination has no directory".into())
        })?;
        let name = destination
            .file_name()
            .ok_or_else(|| AudioExportError::Stream("staged destination has no filename".into()))?;
        let destination = parent.canonicalize()?.join(name);
        require_absent(&destination)?;
        if destination == self.source
            || destination == self.destination
            || self.staged_destination.is_some()
        {
            return Err(AudioExportError::Stream(
                "staged export destination is not independent".into(),
            ));
        }
        self.staged_destination = Some(destination);
        Ok(self)
    }

    /// Encode in a private adjacent file; inspect it before producing publication proof.
    /// Cancellation drops the temporary artifact and kills the owned tool process.
    pub async fn encode(self) -> Result<VerifiedAudioExport, AudioExportError> {
        let input = inspect_stream(&self.source).await?;
        let sample_rate = input.layout.sample_rate.get();
        let channels = input.layout.channels.get();
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
        let parent = self
            .destination
            .parent()
            .ok_or_else(|| AudioExportError::Stream("destination has no directory".into()))?;
        let temporary = tempfile::Builder::new()
            .prefix(".audio-export-")
            .suffix(&format!(".{}", self.format.extension()))
            .tempfile_in(parent)?;
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
        if encoded.layout != input.layout || encoded.codec != self.format.output_codec() {
            return Err(AudioExportError::ProducerLayout);
        }
        let duration = MediaProbe::new(temporary.path()).duration().await?;
        temporary.as_file().sync_all()?;
        let staged = if let Some(destination) = self.staged_destination {
            let staged = tempfile::Builder::new()
                .prefix(".audio-export-")
                .tempfile_in(destination.parent().ok_or_else(|| {
                    AudioExportError::Stream("staged destination has no directory".into())
                })?)?;
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

fn require_absent(destination: &Path) -> Result<(), AudioExportError> {
    match std::fs::symlink_metadata(destination) {
        Ok(_) => Err(AudioExportError::ExistingDestination(
            destination.to_owned(),
        )),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
    }
}

/// Encoded and inspected artifact. No public constructor or replaceable target.
#[derive(Debug)]
pub struct VerifiedAudioExport {
    temporary: NamedTempFile,
    destination: PathBuf,
    duration: AudioDuration,
    staged: Option<StagedExport>,
}

#[derive(Debug)]
struct StagedExport {
    temporary: NamedTempFile,
    destination: PathBuf,
}

impl VerifiedAudioExport {
    /// Measured nonempty encoded duration; not a linguistic-quality claim.
    pub const fn duration(&self) -> AudioDuration {
        self.duration
    }

    /// Publish exactly the admitted destination, atomically refusing existing files.
    pub fn publish(self) -> Result<PathBuf, AudioExportError> {
        verify_destination_parent(&self.destination)?;
        if let Some(staged) = self.staged {
            verify_destination_parent(&staged.destination)?;
            staged
                .temporary
                .persist_noclobber(&staged.destination)
                .map_err(|error| AudioExportError::Io(error.error))?;
        }
        // Publish the user-visible file last. A final competing writer can
        // leave a private staged copy, but never a claimed successful result
        // or an overwritten file; two filesystems are not one transaction.
        self.temporary
            .persist_noclobber(&self.destination)
            .map_err(|error| AudioExportError::Io(error.error))?;
        Ok(self.destination)
    }
}

fn verify_destination_parent(destination: &Path) -> Result<(), AudioExportError> {
    let parent = destination
        .parent()
        .ok_or_else(|| AudioExportError::Stream("export destination has no directory".into()))?;
    if parent.canonicalize()? != parent {
        return Err(AudioExportError::Stream(
            "export destination directory changed after admission".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
