//! File-backed prepared-artifact store for worker protocol V2.
//!
//! Worker protocol V2 splits the boundary into:
//!
//! - a control plane with typed envelopes
//! - a data plane with prepared-artifact descriptors
//!
//! This module owns the data-plane side of worker protocol V2. It provides both
//! a simple file-backed store for prepared audio/text artifacts and a scoped
//! runtime that keeps those artifact files alive for one request family while
//! Rust dispatch code talks to Python workers.

use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

use serde::Serialize;
use thiserror::Error;

use crate::chat_ops::fa::coordinates::FaWindow;
use crate::media::transcode::{PcmEncoding, Transcode, TranscodeError};
use crate::media::window::{DecodedFrames, EmptySegment};
use crate::types::worker_v2::{
    ByteLengthV2, ByteOffsetV2, ChannelCountV2, FrameCountV2, PreparedAudioEncodingV2,
    PreparedAudioRefV2, PreparedTextEncodingV2, PreparedTextRefV2, SampleRateHzV2,
    WorkerArtifactIdV2, WorkerArtifactPathV2,
};

/// Directory names used inside one prepared-artifact root.
const AUDIO_DIR_NAME: &str = "audio";
const TEXT_DIR_NAME: &str = "text";

/// Errors produced while materializing prepared artifacts for protocol V2.
#[derive(Debug, Error)]
pub enum PreparedArtifactErrorV2 {
    /// The transcode itself failed: ffmpeg missing, refused, or unspawnable.
    ///
    /// ONE variant, not the three this enum used to restate. See the matching
    /// note on `EnsureWavError::Transcode`.
    #[error(transparent)]
    Transcode(#[from] TranscodeError),

    /// The requested audio segment yielded no whole samples, which would make
    /// a downstream ML model crash on an empty tensor.
    ///
    /// Callers that encounter this error should skip the group rather than
    /// propagating a hard failure, the utterances will remain unaligned.
    ///
    /// # It does not say the window was past the end of the file
    ///
    /// It used to, in this message and in the one on
    /// `RequestBuildErrorV2::EmptyAudioSegment`, and the claim was never
    /// something the measuring code could know: `ffmpeg` exits 0 and writes
    /// nothing for a window past the end, and it does the same for a window
    /// whose write was truncated. The reason the measurement DID see travels
    /// on the value, so the message states it instead of inventing one.
    ///
    /// This doc also used to say it served "a window that holds nothing" as
    /// well as one that produced nothing. It does not: `FaWindow` refuses a
    /// window whose end is not after its start, so only the second reaches
    /// here.
    #[error("empty audio segment: {0}")]
    EmptyAudioSegment(EmptySegment),

    /// Filesystem or process-management error.
    #[error("prepared artifact I/O error: {0}")]
    Io(#[from] io::Error),
}

/// File-backed prepared-artifact store for protocol V2.
#[derive(Debug, Clone)]
pub struct PreparedArtifactStoreV2 {
    root: PathBuf,
}

impl PreparedArtifactStoreV2 {
    /// Create or open a prepared-artifact store rooted at `root`.
    pub fn new(root: impl AsRef<Path>) -> io::Result<Self> {
        let root = root.as_ref().to_path_buf();
        fs::create_dir_all(root.join(AUDIO_DIR_NAME))?;
        fs::create_dir_all(root.join(TEXT_DIR_NAME))?;
        Ok(Self { root })
    }

    /// Return the filesystem root for this store.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Write mono or multi-channel float32 PCM audio and return the descriptor.
    pub fn write_prepared_audio_f32le(
        &self,
        id: &WorkerArtifactIdV2,
        samples: &[f32],
        sample_rate_hz: SampleRateHzV2,
        channels: ChannelCountV2,
    ) -> io::Result<PreparedAudioRefV2> {
        let channel_count = channels.0 as usize;
        if channel_count == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "prepared audio channels must be greater than zero",
            ));
        }
        if !samples.len().is_multiple_of(channel_count) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "prepared audio sample count must divide evenly by channels",
            ));
        }

        let path = self.audio_path_for(id);
        let file = File::create(&path)?;
        let mut writer = BufWriter::new(file);
        for sample in samples {
            writer.write_all(&sample.to_le_bytes())?;
        }
        writer.flush()?;

        Ok(PreparedAudioRefV2 {
            id: id.clone(),
            path: WorkerArtifactPathV2(path.to_string_lossy().into_owned()),
            encoding: PreparedAudioEncodingV2::PcmF32le,
            channels,
            sample_rate_hz,
            frame_count: FrameCountV2((samples.len() / channel_count) as u64),
            byte_offset: ByteOffsetV2(0),
            byte_len: ByteLengthV2(std::mem::size_of_val(samples) as u64),
        })
    }

    /// Write structured UTF-8 JSON text and return the descriptor.
    pub fn write_prepared_text_json<T: Serialize>(
        &self,
        id: &WorkerArtifactIdV2,
        value: &T,
    ) -> io::Result<PreparedTextRefV2> {
        let path = self.text_path_for(id);
        let bytes = serde_json::to_vec(value)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error))?;
        fs::write(&path, &bytes)?;

        Ok(PreparedTextRefV2 {
            id: id.clone(),
            path: WorkerArtifactPathV2(path.to_string_lossy().into_owned()),
            encoding: PreparedTextEncodingV2::Utf8Json,
            byte_offset: ByteOffsetV2(0),
            byte_len: ByteLengthV2(bytes.len() as u64),
        })
    }

    /// Extract a mono 16 kHz float32 PCM audio span using ffmpeg.
    ///
    /// This is the first concrete data-plane step toward worker protocol V2:
    /// Rust materializes the model-ready audio span up front, and the future
    /// Python worker will consume the resulting descriptor rather than reading
    /// and chunking source media on its own.
    pub async fn extract_prepared_audio_segment_f32le(
        &self,
        id: &WorkerArtifactIdV2,
        source: &Path,
        window: FaWindow,
    ) -> Result<PreparedAudioRefV2, PreparedArtifactErrorV2> {
        let root = self.root.clone();
        let id = id.clone();
        let source = source.to_path_buf();

        crate::blocking::spawn_in_span(move || {
            fs::create_dir_all(root.join(AUDIO_DIR_NAME))?;
            let output_path = root.join(AUDIO_DIR_NAME).join(format!("{id}.pcm"));

            let produced =
                Transcode::window(&source, window, PcmEncoding::F32LeRaw).produce(&output_path)?;
            let byte_len = produced.byte_len();

            // ffmpeg exits with code 0 and writes an empty (or truncated)
            // output rather than failing, and an empty tensor crashes the
            // downstream models with an opaque kernel-size error. So the
            // measurement is the gate, and it is THIS function's job: nothing
            // upstream can see the byte length and nothing downstream can.
            //
            // The reason travels with the failure. It used to be `frame_count
            // == 0` and an error built from the window alone, which threw away
            // the one fact only this line could observe; `error.rs` then
            // documented a reason nobody here had witnessed.
            let frame_count = match DecodedFrames::measure_f32le(byte_len) {
                DecodedFrames::Frames(frames) => frames,
                DecodedFrames::None(reason) => {
                    let _ = fs::remove_file(&output_path);
                    return Err(PreparedArtifactErrorV2::EmptyAudioSegment(EmptySegment {
                        path: source.display().to_string(),
                        window,
                        reason,
                    }));
                }
            };

            Ok(PreparedAudioRefV2 {
                id,
                path: WorkerArtifactPathV2(output_path.to_string_lossy().into_owned()),
                encoding: PreparedAudioEncodingV2::PcmF32le,
                // Derived from what was produced, not restated. These were
                // two literals mirroring constants in another module.
                channels: ChannelCountV2(produced.channels()),
                sample_rate_hz: SampleRateHzV2(produced.sample_rate_hz()),
                frame_count: FrameCountV2(frame_count.get()),
                byte_offset: ByteOffsetV2(0),
                byte_len: ByteLengthV2(byte_len),
            })
        })
        .await
        .map_err(io::Error::other)?
    }

    /// Convert a full media file into a mono 16 kHz float32 PCM artifact.
    ///
    /// This is the whole-file counterpart to
    /// [`extract_prepared_audio_segment_f32le`](Self::extract_prepared_audio_segment_f32le).
    /// It is used when Rust wants to own ASR audio preparation but the model
    /// should still see the entire file instead of one explicit subspan.
    pub async fn prepare_audio_file_f32le(
        &self,
        id: &WorkerArtifactIdV2,
        source: &Path,
    ) -> Result<PreparedAudioRefV2, PreparedArtifactErrorV2> {
        let root = self.root.clone();
        let id = id.clone();
        let source = source.to_path_buf();

        crate::blocking::spawn_in_span(move || {
            fs::create_dir_all(root.join(AUDIO_DIR_NAME))?;
            let output_path = root.join(AUDIO_DIR_NAME).join(format!("{id}.pcm"));

            let produced =
                Transcode::whole(&source, PcmEncoding::F32LeRaw).produce(&output_path)?;
            let byte_len = produced.byte_len();
            let sample_bytes = std::mem::size_of::<f32>() as u64;
            let frame_count = byte_len / sample_bytes;

            Ok(PreparedAudioRefV2 {
                id,
                path: WorkerArtifactPathV2(output_path.to_string_lossy().into_owned()),
                encoding: PreparedAudioEncodingV2::PcmF32le,
                // Derived from what was produced, not restated. These were
                // two literals mirroring constants in another module.
                channels: ChannelCountV2(produced.channels()),
                sample_rate_hz: SampleRateHzV2(produced.sample_rate_hz()),
                frame_count: FrameCountV2(frame_count),
                byte_offset: ByteOffsetV2(0),
                byte_len: ByteLengthV2(byte_len),
            })
        })
        .await
        .map_err(io::Error::other)?
    }

    /// Convert verified in-memory media bytes into mono 16 kHz float32 PCM.
    ///
    /// This variant is for cache-authorized provider evidence. The caller has
    /// already proved the bytes against the cache identity, so materializing a
    /// private temporary source prevents a mutable external path from changing
    /// between authorization and ffmpeg's read.
    pub async fn prepare_audio_bytes_f32le(
        &self,
        id: &WorkerArtifactIdV2,
        source_bytes: Vec<u8>,
    ) -> Result<PreparedAudioRefV2, PreparedArtifactErrorV2> {
        let root = self.root.clone();
        let id = id.clone();

        crate::blocking::spawn_in_span(move || {
            fs::create_dir_all(root.join(AUDIO_DIR_NAME))?;
            let mut source = tempfile::Builder::new()
                .prefix("verified-source-")
                .suffix(".media")
                .tempfile_in(&root)?;
            source.write_all(&source_bytes)?;
            source.flush()?;

            let output_path = root.join(AUDIO_DIR_NAME).join(format!("{id}.pcm"));
            let produced =
                Transcode::whole(source.path(), PcmEncoding::F32LeRaw).produce(&output_path)?;
            let byte_len = produced.byte_len();
            let sample_bytes = std::mem::size_of::<f32>() as u64;
            let frame_count = byte_len / sample_bytes;

            Ok(PreparedAudioRefV2 {
                id,
                path: WorkerArtifactPathV2(output_path.to_string_lossy().into_owned()),
                encoding: PreparedAudioEncodingV2::PcmF32le,
                channels: ChannelCountV2(produced.channels()),
                sample_rate_hz: SampleRateHzV2(produced.sample_rate_hz()),
                frame_count: FrameCountV2(frame_count),
                byte_offset: ByteOffsetV2(0),
                byte_len: ByteLengthV2(byte_len),
            })
        })
        .await
        .map_err(io::Error::other)?
    }

    /// Return the canonical audio path for one artifact id.
    fn audio_path_for(&self, id: &WorkerArtifactIdV2) -> PathBuf {
        self.root.join(AUDIO_DIR_NAME).join(format!("{id}.pcm"))
    }

    /// Return the canonical text path for one artifact id.
    fn text_path_for(&self, id: &WorkerArtifactIdV2) -> PathBuf {
        self.root.join(TEXT_DIR_NAME).join(format!("{id}.json"))
    }
}

/// Scoped prepared-artifact runtime for one V2 worker dispatch seam.
///
/// V2 request builders return descriptors that point at files on disk. The
/// runtime owns the temporary directory that backs those files and exposes the
/// regular [`PreparedArtifactStoreV2`] API for materialization, ensuring the
/// descriptors stay valid until the enclosing request/response flow completes.
#[derive(Debug)]
pub(crate) struct PreparedArtifactRuntimeV2 {
    _scratch_dir: tempfile::TempDir,
    store: PreparedArtifactStoreV2,
}

impl PreparedArtifactRuntimeV2 {
    /// Create a fresh scoped runtime rooted under a new temporary directory.
    pub(crate) fn new(scope_name: impl Into<PathBuf>) -> io::Result<Self> {
        let scratch_dir = tempfile::tempdir()?;
        let store = PreparedArtifactStoreV2::new(scratch_dir.path().join(scope_name.into()))?;
        Ok(Self {
            _scratch_dir: scratch_dir,
            store,
        })
    }

    /// Borrow the file-backed store owned by this runtime.
    pub(crate) const fn store(&self) -> &PreparedArtifactStoreV2 {
        &self.store
    }

    /// The artifact directory while this runtime owns it.
    #[cfg(test)]
    pub(crate) fn root(&self) -> &Path {
        self.store.root()
    }
}

#[cfg(test)]
mod tests;
