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

use crate::media::transcode::{PcmEncoding, Transcode, TranscodeError};
use crate::media::window::{DecodedFrames, EmptySegment, MediaWindow};
use crate::types::worker_v2::{
    ArtifactRefV2, ByteLengthV2, ByteOffsetV2, ChannelCountV2, FrameCountV2,
    PreparedAudioEncodingV2, PreparedAudioRefV2, PreparedTextEncodingV2, PreparedTextRefV2,
    SampleRateHzV2, WorkerArtifactIdV2, WorkerArtifactPathV2,
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
    /// well as one that produced nothing. It does not: `MediaWindow::new`
    /// refuses a window whose end is not after its start, so the first of
    /// those two is unconstructible and only the second reaches here.
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
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Create an inline JSON attachment without touching the filesystem.
    pub fn inline_json(&self, id: WorkerArtifactIdV2, value: serde_json::Value) -> ArtifactRefV2 {
        ArtifactRefV2::InlineJson(crate::types::worker_v2::InlineJsonRefV2 { id, value })
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
        window: MediaWindow,
    ) -> Result<PreparedAudioRefV2, PreparedArtifactErrorV2> {
        let root = self.root.clone();
        let id = id.clone();
        let source = source.to_path_buf();

        tokio::task::spawn_blocking(move || {
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

        tokio::task::spawn_blocking(move || {
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

        tokio::task::spawn_blocking(move || {
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
    pub(crate) fn store(&self) -> &PreparedArtifactStoreV2 {
        &self.store
    }

    /// Return the store root for diagnostics and tests.
    #[cfg(test)]
    pub(crate) fn root(&self) -> &Path {
        self.store.root()
    }

    /// Return the outer scratch directory kept alive by this runtime.
    #[cfg(test)]
    pub(crate) fn scratch_root(&self) -> &Path {
        self._scratch_dir.path()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::media::tools::MediaTool;
    // Named only here: the production path receives a reason from
    // `DecodedFrames::measure_f32le` and never spells the type.
    use crate::media::window::EmptyReason;
    use crate::time::FileMs;

    /// The reason a decode of `byte_len` bytes reports, FROM THE MEASUREMENT.
    ///
    /// The only way a test can obtain an [`EmptyReason`], and since 2026-09-07
    /// the only way anything can: the variants were public, and the message
    /// test below wrote the struct literal, so it asserted a reason nothing had
    /// measured and would have gone on passing after the measurement stopped
    /// producing that reason at all. Going through the producer makes the same
    /// test evidence about the pair.
    fn measured_reason(byte_len: u64) -> EmptyReason {
        match DecodedFrames::measure_f32le(byte_len) {
            DecodedFrames::None(reason) => reason,
            DecodedFrames::Frames(frames) => {
                panic!("expected an empty decode of {byte_len} bytes, got {frames} whole frames")
            }
        }
    }

    /// Create a temporary prepared-artifact store for unit tests.
    fn test_store() -> (PreparedArtifactStoreV2, tempfile::TempDir) {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = PreparedArtifactStoreV2::new(dir.path()).expect("prepared artifact store");
        (store, dir)
    }

    /// Create a scoped prepared-artifact runtime for unit tests.
    fn test_runtime() -> PreparedArtifactRuntimeV2 {
        PreparedArtifactRuntimeV2::new("artifacts").expect("prepared artifact runtime")
    }

    #[test]
    fn creates_expected_directory_layout() {
        let runtime = test_runtime();
        assert!(runtime.root().join(AUDIO_DIR_NAME).is_dir());
        assert!(runtime.root().join(TEXT_DIR_NAME).is_dir());
        assert!(runtime.scratch_root().is_dir());
    }

    #[test]
    fn writes_prepared_audio_descriptor_and_pcm_bytes() {
        let (store, _dir) = test_store();
        let id = WorkerArtifactIdV2::from("audio-ref-1");
        let descriptor = store
            .write_prepared_audio_f32le(
                &id,
                &[0.25, -0.5, 1.0, 0.0],
                SampleRateHzV2(16_000),
                ChannelCountV2(2),
            )
            .expect("write prepared audio");

        assert_eq!(descriptor.id, id);
        assert_eq!(descriptor.channels, 2);
        assert_eq!(descriptor.frame_count, 2);
        assert_eq!(descriptor.byte_len, 16);

        let bytes = fs::read(Path::new(descriptor.path.as_ref())).expect("read prepared pcm");
        let expected: Vec<u8> = [0.25_f32, -0.5_f32, 1.0_f32, 0.0_f32]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        assert_eq!(bytes, expected);
    }

    #[test]
    fn rejects_audio_when_sample_count_does_not_match_channels() {
        let (store, _dir) = test_store();
        let error = store
            .write_prepared_audio_f32le(
                &WorkerArtifactIdV2::from("audio-ref-2"),
                &[0.1, 0.2, 0.3],
                SampleRateHzV2(16_000),
                ChannelCountV2(2),
            )
            .expect_err("misaligned sample count should fail");

        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
    }

    #[test]
    fn writes_prepared_text_descriptor_and_json_bytes() {
        let (store, _dir) = test_store();
        let id = WorkerArtifactIdV2::from("text-ref-1");
        let descriptor = store
            .write_prepared_text_json(&id, &serde_json::json!({"words": ["hello", "world"]}))
            .expect("write prepared text");

        assert_eq!(descriptor.id, id);
        assert_eq!(descriptor.encoding, PreparedTextEncodingV2::Utf8Json);

        let raw = fs::read_to_string(Path::new(descriptor.path.as_ref())).expect("read text");
        let parsed: serde_json::Value = serde_json::from_str(&raw).expect("parse text json");
        assert_eq!(parsed, serde_json::json!({"words": ["hello", "world"]}));
        assert_eq!(descriptor.byte_offset, 0);
        assert_eq!(descriptor.byte_len, raw.len() as u64);
    }

    #[tokio::test]
    async fn extracts_prepared_audio_segment_with_ffmpeg() {
        if MediaTool::Ffmpeg.banner().is_none() {
            eprintln!("skipping: ffmpeg not installed");
            return;
        }

        let (store, dir) = test_store();
        let wav_path = dir.path().join("tone.wav");
        let ffmpeg_out = MediaTool::Ffmpeg
            .async_command()
            .args([
                "-y",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:sample_rate=16000",
                "-t",
                "0.25",
                wav_path.to_string_lossy().as_ref(),
            ])
            .output()
            .await
            .expect("ffmpeg process should run");
        if !ffmpeg_out.status.success() {
            eprintln!("skipping: could not generate test wav");
            return;
        }

        let descriptor = store
            .extract_prepared_audio_segment_f32le(
                &WorkerArtifactIdV2::from("audio-segment-ref"),
                &wav_path,
                MediaWindow::new(FileMs::new(0), FileMs::new(100)).expect("non-empty"),
            )
            .await
            .expect("extract prepared audio segment");

        assert_eq!(descriptor.channels, 1);
        assert_eq!(descriptor.sample_rate_hz, 16_000);
        assert!(descriptor.frame_count.0 > 0);
        assert!(descriptor.byte_len.0 > 0);
        assert!(Path::new(descriptor.path.as_ref()).exists());
    }

    /// Helper: generate a short tone WAV for artifact tests.
    async fn write_test_tone_wav(path: &std::path::Path, duration_s: f32) {
        let out = MediaTool::Ffmpeg
            .async_command()
            .args([
                "-y",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:sample_rate=16000",
                "-t",
                &format!("{duration_s}"),
                path.to_string_lossy().as_ref(),
            ])
            .output()
            .await
            .expect("ffmpeg process should run");
        assert!(
            out.status.success(),
            "ffmpeg should generate the artifact test tone: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    /// Regression test for the Croatian/PWA empty-audio-segment crash.
    ///
    /// When an FA group's timestamps extend past the end of the audio file,
    /// ffmpeg exits with code 0 but writes zero bytes.  Before the fix, this
    /// produced a 0-frame descriptor that caused Wave2Vec to crash with
    /// "Kernel size can't be greater than actual input size".  After the fix,
    /// the extractor returns `PreparedArtifactErrorV2::EmptyAudioSegment`.
    #[tokio::test]
    async fn extract_returns_empty_audio_segment_error_when_segment_past_end_of_file() {
        if MediaTool::Ffmpeg.banner().is_none() {
            eprintln!("skipping: ffmpeg not installed");
            return;
        }

        let (store, dir) = test_store();
        let wav_path = dir.path().join("short.wav");
        // Generate a 0.1-second WAV, valid range is 0..100ms.
        write_test_tone_wav(&wav_path, 0.1).await;

        // Request a segment (500ms..600ms) entirely past the end of the file.
        let result = store
            .extract_prepared_audio_segment_f32le(
                &WorkerArtifactIdV2::from("empty-audio-test"),
                &wav_path,
                MediaWindow::new(FileMs::new(500), FileMs::new(600)).expect("non-empty"),
            )
            .await;

        let Err(PreparedArtifactErrorV2::EmptyAudioSegment(segment)) = result else {
            panic!("expected EmptyAudioSegment error, got: {result:?}");
        };
        // The window travels whole, so the test can assert it as one value
        // rather than pattern-matching two loose integers out of the variant.
        assert_eq!(segment.window.start().get(), 500);
        assert_eq!(segment.window.end().get(), 600);
        // And the reason is the one the measurement actually saw. This is the
        // real past-the-end case, so ffmpeg writes nothing at all; the value
        // says that rather than the message asserting it, and the expectation
        // comes from the same measurement rather than from a literal.
        assert_eq!(segment.reason, measured_reason(0));
    }

    /// The message states the reason that was measured and does not assert a
    /// cause the measurement cannot know.
    ///
    /// This variant's message, and the one on
    /// `RequestBuildErrorV2::EmptyAudioSegment`, both used to end with "past
    /// the end of the file". A truncated write produces the same zero whole
    /// frames from a window well inside the file, so the claim was a guess
    /// wearing the grammar of a fact. A partial frame is the case that proves
    /// it: bytes WERE written, so the segment plainly was not past the end.
    #[test]
    fn an_empty_segment_reports_what_was_measured_not_a_guessed_cause() {
        let window = MediaWindow::new(FileMs::new(0), FileMs::new(100))
            .expect("a window that ends after it starts");
        let rendered = PreparedArtifactErrorV2::EmptyAudioSegment(EmptySegment {
            path: "clip.wav".to_owned(),
            window,
            // Three bytes of a four-byte sample, obtained by MEASURING three
            // bytes rather than by asserting what that measurement would say.
            reason: measured_reason(3),
        })
        .to_string();

        assert!(
            rendered.contains("less than one 4-byte sample"),
            "the message should carry what the decoder wrote: {rendered}"
        );
        assert!(
            !rendered.contains("past the end"),
            "the message must not assert a cause nothing measured: {rendered}"
        );
    }

    #[test]
    fn builds_inline_json_attachment_without_filesystem_write() {
        let (store, dir) = test_store();
        let attachment = store.inline_json(
            WorkerArtifactIdV2::from("inline-ref-1"),
            serde_json::json!({"utterances": [{"words": ["hi"]}]}),
        );

        let ArtifactRefV2::InlineJson(inline) = attachment else {
            panic!("expected inline json attachment");
        };
        assert_eq!(inline.id, "inline-ref-1");
        assert_eq!(
            inline.value,
            serde_json::json!({"utterances": [{"words": ["hi"]}]})
        );

        let files: Vec<_> = walkdir::WalkDir::new(dir.path())
            .min_depth(1)
            .into_iter()
            .filter_map(Result::ok)
            .filter(|entry| entry.file_type().is_file())
            .collect();
        assert!(
            files.is_empty(),
            "inline attachments should not create files"
        );
    }
}
