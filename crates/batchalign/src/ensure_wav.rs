//! Content-fingerprinted WAV conversion cache.
//!
//! Inspects actual containers and converts media requiring native-decoder
//! preparation to WAV via ffmpeg. It caches the result
//! in `~/.batchalign3/media_cache/` keyed by a complete source-content digest. This
//! avoids writing converted files next to originals (which may be on read-only
//! NFS/SMB mounts) and deduplicates across paths that point to the same content.
//!
//!
//! # Design
//!
//! - **Identity**: full streamed BLAKE3 of source bytes in a versioned PCM
//!   recipe namespace. Legacy sampled-fingerprint entries remain on disk but
//!   are not silently admitted as complete-content matches.
//! - **Locking**: per-fingerprint `.lock` file via std `File::lock` (exclusive) prevents
//!   concurrent ffmpeg invocations for the same source file (important for
//!   parallel FA groups).
//! - **Atomic writes**: convert to a temp file, then `rename()` into the cache.
//!
//! # Formats requiring conversion
//!
//! ffprobe admits the actual audio container before selecting a path. Native
//! MPEG audio, WAVE, FLAC and Ogg retain their source path. ISO media, ADTS,
//! Matroska and ASF use the PCM cache regardless of their filename suffix.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use thiserror::Error;
use tracing::{debug, info};

use crate::chat_ops::fa::coordinates::FaWindow;
use crate::media::probe::{
    AudioDuration, MediaContainer, MediaProbe, ObservedMediaHeader, ProbeError,
};
use crate::media::transcode::{PcmEncoding, Transcode, TranscodeError};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// A source whose observed container admits one preparation operation.
/// Private payload constructors bind that observation to its own source path.
enum MediaPreparation {
    Native(NativeContainerSource),
    Convert(ConversionSource),
}

/// Only native-container admission can produce a source for passthrough.
struct NativeContainerSource(ObservedMediaHeader);

/// Only conversion admission can produce a source for the PCM cache.
struct ConversionSource(ObservedMediaHeader);

impl MediaPreparation {
    async fn inspect(source: PathBuf) -> Result<Self, ProbeError> {
        let header = MediaProbe::new(source).observe().await?;
        match header.container() {
            MediaContainer::MpegAudio
            | MediaContainer::Wav
            | MediaContainer::Flac
            | MediaContainer::Ogg => Ok(Self::Native(NativeContainerSource(header))),
            MediaContainer::Adts
            | MediaContainer::IsoMedia
            | MediaContainer::Matroska
            | MediaContainer::Asf => Ok(Self::Convert(ConversionSource(header))),
        }
    }

    async fn prepare(self, cache_dir: Option<&Path>) -> Result<PreparedMedia, EnsureWavError> {
        Ok(match self {
            Self::Native(source) => PreparedMedia::Native(source),
            Self::Convert(source) => PreparedMedia::Converted(source.convert(cache_dir).await?),
        })
    }
}

impl NativeContainerSource {
    fn into_path(self) -> PathBuf {
        self.0.into_path()
    }

    async fn measure(self) -> MeasuredAudio {
        let (path, outcome) = self.0.measure().await;
        MeasuredAudio::new(path, outcome)
    }
}

/// A WAV in the conversion cache.
enum ConvertedMedia {
    /// Converted by this call: its length was read right after conversion.
    Fresh {
        path: PathBuf,
        duration: Result<AudioDuration, ProbeError>,
    },
    /// Already published by an earlier conversion; measured when asked.
    Published(PathBuf),
}

enum PreparedMedia {
    Native(NativeContainerSource),
    Converted(ConvertedMedia),
}

impl PreparedMedia {
    fn into_path(self) -> PathBuf {
        match self {
            Self::Native(native) => native.into_path(),
            Self::Converted(ConvertedMedia::Fresh { path, .. })
            | Self::Converted(ConvertedMedia::Published(path)) => path,
        }
    }

    async fn measure(self) -> MeasuredAudio {
        match self {
            Self::Native(native) => native.measure().await,
            Self::Converted(ConvertedMedia::Fresh { path, duration }) => {
                MeasuredAudio::new(path, duration)
            }
            Self::Converted(ConvertedMedia::Published(path)) => MeasuredAudio::probe(path).await,
        }
    }
}

/// The prepared path and its completed measurement attempt. Refusal is a
/// terminal observation for this owner and can be shared across FA and UTR.
pub struct MeasuredAudio {
    path: PathBuf,
    outcome: Result<AudioDuration, Arc<ProbeError>>,
}

impl MeasuredAudio {
    fn new(path: PathBuf, outcome: Result<AudioDuration, ProbeError>) -> Self {
        Self {
            path,
            outcome: outcome.map_err(Arc::new),
        }
    }

    /// Measure a path preparation produced but did not measure. Private:
    /// a public constructor let any caller hand the FA worker a path that
    /// never passed container admission (an mp4 soundfile cannot read), and
    /// the failure surfaced inside the worker instead of at admission.
    /// Production reaches it only through [`prepare_measured_audio`].
    async fn probe(path: impl Into<PathBuf>) -> Self {
        let path = path.into();
        let outcome = MediaProbe::new(&path).duration().await;
        Self::new(path, outcome)
    }

    /// A real measurement of a native fixture, for tests that build an FA
    /// context around a recording they wrote themselves.
    #[cfg(test)]
    pub(crate) async fn probe_fixture(path: impl Into<PathBuf>) -> Self {
        Self::probe(path).await
    }

    /// Pure fixture for a downstream check that owns its window separately.
    /// Deliberately refuses if the check tries to acquire a recording here.
    #[cfg(test)]
    pub(crate) fn unmeasurable_fixture(path: impl Into<PathBuf>) -> Self {
        let path = path.into();
        let error = ProbeError::Refused {
            input: path.display().to_string(),
            diagnostics: "the fixture supplies its own window".to_owned(),
        };
        Self::new(path, Err(error))
    }

    /// The audio whose timeline was observed.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Read the same admitted duration or refusal without another probe.
    pub const fn duration(&self) -> Result<AudioDuration, &Arc<ProbeError>> {
        self.outcome.as_ref().copied()
    }
}

/// Default cache directory name under the app data dir.
const CACHE_SUBDIR: &str = "media_cache";

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
/// Errors from [`ensure_wav`].
pub enum EnsureWavError {
    /// Actual container admission failed before selecting a decoder path.
    #[error(transparent)]
    Probe(#[from] ProbeError),

    /// The transcode itself failed: ffmpeg missing, refused, or unspawnable.
    ///
    /// ONE variant, not the three this enum used to restate. Nothing in the
    /// crate ever matched on `FfmpegNotFound` or `FfmpegFailed`; they existed
    /// only to re-word `TranscodeError`'s own Display text, which is five
    /// statements of one reading and a sixth owed by the next consumer.
    #[error(transparent)]
    Transcode(#[from] TranscodeError),

    /// Filesystem I/O error.
    #[error("I/O error during media conversion: {0}")]
    Io(#[from] std::io::Error),
}

// ---------------------------------------------------------------------------
// The cache slot
// ---------------------------------------------------------------------------

/// What acquiring a cache slot found.
enum CacheSlot {
    /// Another writer published while we waited for the lock.
    AlreadyPublished(PathBuf),
    /// Ours to write, under the lock we now hold.
    Ours(LockedCacheSlot),
}

/// A cache entry held under an exclusive per-key lock.
///
/// # Why this is a type and not two lines of code
///
/// Consuming this slot through `produce` yields a `ProducedCacheSlot` only
/// after strict conversion into this slot's own temporary path. Only that
/// resulting state can commit, while still holding the original lock. No
/// caller can supply an unrelated conversion token to publish this slot.
/// Private fields enforce the boundary outside this module; filesystem
/// mutation by unrelated processes remains outside this guarantee.
///
/// The lock is held from the re-check through the rename that publishes:
/// two callers converting the same fingerprint share one `{key}.tmp.wav`,
/// and a lock released before the rename would let the second write that
/// temp path while the first renames it into the cache, which the
/// fingerprinted fast path would then serve forever.
struct LockedCacheSlot {
    /// Dropping this releases the lock, which is why `commit` renames first.
    lock: crate::file_lock::HeldFileLock,
    tmp_path: PathBuf,
    cached_path: PathBuf,
}

impl LockedCacheSlot {
    /// Take the lock for `key`, or report that someone else got there first.
    fn acquire(cache_dir: &Path, key: &str) -> Result<CacheSlot, std::io::Error> {
        let cached_path = cache_dir.join(format!("{key}.wav"));
        let lock =
            crate::file_lock::HeldFileLock::acquire(cache_dir.join(format!("{key}.wav.lock")))?;
        // Re-check under the lock: another writer may have finished while we
        // waited. Now that publishing happens under the lock too, this answer
        // cannot go stale between the check and the caller acting on it.
        if cached_path.exists() {
            return Ok(CacheSlot::AlreadyPublished(cached_path));
        }
        Ok(CacheSlot::Ours(Self {
            lock,
            tmp_path: cache_dir.join(format!("{key}.tmp.wav")),
            cached_path,
        }))
    }

    /// Convert into this exact slot, retaining its lock across the transition.
    fn produce<T>(
        self,
        write: impl FnOnce(&Path) -> Result<T, TranscodeError>,
    ) -> Result<ProducedCacheSlot<T>, TranscodeError> {
        let produced = write(&self.tmp_path)?;
        Ok(ProducedCacheSlot {
            slot: self,
            produced,
        })
    }
}

/// A locked slot whose own temporary destination passed strict conversion,
/// with what the conversion reported.
struct ProducedCacheSlot<T> {
    slot: LockedCacheSlot,
    produced: T,
}

impl<T> ProducedCacheSlot<T> {
    /// Publish atomically before releasing the lock; hand back the published
    /// path and the conversion's report.
    fn commit(self) -> Result<(PathBuf, T), std::io::Error> {
        let LockedCacheSlot {
            lock,
            tmp_path,
            cached_path,
        } = self.slot;
        std::fs::rename(&tmp_path, &cached_path)?;
        // Explicit, and AFTER the rename: this order is the entire point of the
        // type, so it is stated rather than left to end-of-scope drop order.
        drop(lock);
        Ok((cached_path, self.produced))
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Prepare media for the native audio decoder using its observed container.
///
/// Native audio containers retain their source path. Other admitted containers
/// use the existing content-addressed PCM WAV cache. Filename suffixes do not
/// grant passthrough, and unreadable media is refused during admission.
/// `cache_dir` overrides the default cache location for an owned runtime or test.
pub async fn ensure_wav(
    source_path: &Path,
    cache_dir: Option<&Path>,
) -> Result<PathBuf, EnsureWavError> {
    Ok(prepare_media(source_path, cache_dir).await?.into_path())
}

/// Prepare the path used by FA and settle its duration attempt exactly once.
pub(crate) async fn prepare_measured_audio(source: &Path) -> Result<MeasuredAudio, EnsureWavError> {
    Ok(prepare_media(source, None).await?.measure().await)
}

async fn prepare_media(
    source: &Path,
    cache_dir: Option<&Path>,
) -> Result<PreparedMedia, EnsureWavError> {
    MediaPreparation::inspect(source.to_path_buf())
        .await?
        .prepare(cache_dir)
        .await
}

impl ConversionSource {
    async fn convert(self, cache_dir: Option<&Path>) -> Result<ConvertedMedia, EnsureWavError> {
        let source = self.0;
        let cache_dir = cache_dir.map(Path::to_path_buf);

        crate::blocking::spawn_in_span(move || -> Result<ConvertedMedia, EnsureWavError> {
            let source_path = source.path();
            let effective_cache_dir = match cache_dir {
                Some(p) => p,
                None => default_cache_dir(),
            };
            std::fs::create_dir_all(&effective_cache_dir)?;

            let fp = media_fingerprint(source_path)?;
            let cached_wav = effective_cache_dir.join(format!("{fp}.wav"));

            // Fast path: already converted
            if cached_wav.exists() {
                debug!(
                    "Media cache hit: {} -> {}",
                    source_path.display(),
                    cached_wav.display()
                );
                return Ok(ConvertedMedia::Published(cached_wav));
            }

            // Convert under a per-fingerprint exclusive lock, held until the result
            // is published, which `LockedCacheSlot` is what guarantees.
            let slot = match LockedCacheSlot::acquire(&effective_cache_dir, &fp)? {
                CacheSlot::AlreadyPublished(path) => {
                    return Ok(ConvertedMedia::Published(path));
                }
                CacheSlot::Ours(slot) => slot,
            };

            info!(
                "Converting {} -> {}",
                source_path.display(),
                cached_wav.display()
            );

            // The status check, the partial-file cleanup and the stderr reading all
            // live in `produce`; this call site keeps only what is its own.
            let (path, wav) = slot
                .produce(|destination| Transcode::produce_observed_wav(source, destination))?
                .commit()?;
            Ok(ConvertedMedia::Fresh {
                path,
                duration: wav.into_duration(),
            })
        })
        .await
        .map_err(std::io::Error::other)?
    }
}

/// Extract a time-windowed audio segment to a cached WAV file.
///
/// Uses ffmpeg `-ss {start} -to {end}` with `pcm_s16le` 16kHz mono.
/// Result is cached in `media_cache` keyed by source fingerprint + time window.
///
/// Takes the recording-bound [`FaWindow`], so a damaged decode is judged
/// against the window's own length instead of a probe of the whole source. See
/// `Transcode::window` for what the window's proof does not cover.
pub async fn extract_audio_segment(
    source: &Path,
    window: FaWindow,
) -> Result<PathBuf, EnsureWavError> {
    let source = source.to_path_buf();
    // Only for the cache key: the window itself is already proven, so nothing
    // here re-checks it. That check used to live in this function, and in two
    // others, none of them where the pair originates.
    let (start_ms, end_ms) = (window.audio_start().get(), window.end().get());

    crate::blocking::spawn_in_span(move || -> Result<PathBuf, EnsureWavError> {
        let cache_dir = default_cache_dir();
        std::fs::create_dir_all(&cache_dir)?;

        // Key includes source identity + time window
        // Failure to identify the source is a refusal, never a cache key
        // synthesized from its length (or zero when metadata also failed).
        let base_fp = media_fingerprint(&source)?;
        let segment_key = format!(
            "segment-pcm16-mono16k-strict-v3-{}",
            blake3::hash(format!("{base_fp}|{start_ms}|{end_ms}").as_bytes()).to_hex()
        );
        let cached_wav = cache_dir.join(format!("{segment_key}.wav"));

        // Fast path: already extracted
        if cached_wav.exists() {
            debug!(
                "Segment cache hit: {}[{}..{}ms] -> {}",
                source.display(),
                start_ms,
                end_ms,
                cached_wav.display()
            );
            return Ok(cached_wav);
        }

        // Convert under lock, held until the result is published.
        let slot = match LockedCacheSlot::acquire(&cache_dir, &segment_key)? {
            CacheSlot::AlreadyPublished(path) => return Ok(path),
            CacheSlot::Ours(slot) => slot,
        };

        let start_secs = start_ms as f64 / 1000.0;
        let end_secs = end_ms as f64 / 1000.0;

        info!(
            "Extracting segment {}[{:.3}s..{:.3}s] -> {}",
            source.display(),
            start_secs,
            end_secs,
            cached_wav.display()
        );

        let transcode = Transcode::window(&source, window, PcmEncoding::S16LeWav);
        let produced = slot.produce(|destination| transcode.produce(destination))?;
        Ok(produced.commit()?.0)
    })
    .await
    .map_err(std::io::Error::other)?
}

/// Remove all files in the media cache. Returns bytes freed.
pub fn clear_cache(cache_dir: Option<&Path>) -> std::io::Result<u64> {
    let dir = match cache_dir {
        Some(p) => p.to_path_buf(),
        None => default_cache_dir(),
    };
    if !dir.is_dir() {
        return Ok(0);
    }
    let mut freed = 0u64;
    for entry in std::fs::read_dir(&dir)? {
        let entry = entry?;
        let path = entry.path();
        if let Ok(meta) = std::fs::metadata(&path) {
            freed += meta.len();
            let _ = std::fs::remove_file(&path);
        }
    }
    Ok(freed)
}

/// Return `(cache_dir, total_size_bytes, file_count)`.
#[must_use]
pub fn cache_stats(cache_dir: Option<&Path>) -> (PathBuf, u64, usize) {
    let dir = match cache_dir {
        Some(p) => p.to_path_buf(),
        None => default_cache_dir(),
    };
    if !dir.is_dir() {
        return (dir, 0, 0);
    }
    let mut total_size = 0u64;
    let mut file_count = 0usize;
    if let Ok(entries) = std::fs::read_dir(&dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|e| e == "wav")
                && let Ok(meta) = std::fs::metadata(&path)
            {
                total_size += meta.len();
                file_count += 1;
            }
        }
    }
    (dir, total_size, file_count)
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

/// Complete source identity, namespaced by the whole-file conversion recipe.
///
/// A sampled fingerprint cannot attest content: equal-sized files may differ
/// exclusively between their first and last blocks. Streaming bounds memory,
/// while hashing every byte makes such changes select distinct cache entries.
fn media_fingerprint(path: &Path) -> Result<String, std::io::Error> {
    let file = std::fs::File::open(path)?;
    let mut hasher = blake3::Hasher::new();
    hasher.update_reader(file)?;
    Ok(format!(
        "whole-pcm16-mono16k-strict-v3-{}",
        hasher.finalize().to_hex()
    ))
}

fn default_cache_dir() -> PathBuf {
    default_cache_dir_from(
        crate::runtime_paths::media_cache_dir_override_from_env(),
        crate::runtime_paths::BatchalignHome::from_env().data_base(),
    )
}

fn default_cache_dir_from(
    override_dir: Option<PathBuf>,
    platform_data_dir: Option<PathBuf>,
) -> PathBuf {
    override_dir.unwrap_or_else(|| {
        platform_data_dir
            .unwrap_or_else(|| PathBuf::from("."))
            .join("batchalign3")
            .join(CACHE_SUBDIR)
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::media::tools::MediaTool;

    /// Guards: The versioned cache key covers the complete file, independent
    /// of path, including same-size changes between earlier sampled blocks.
    /// Kind: contract
    /// Not a type: Different native files can contain identical bytes, and
    /// file content can change while keeping exactly the same length.
    /// State: Two paths own one three-block source and a same-size mutation.
    /// Concurrency: The fixture is the only producer of both files.
    #[test]
    fn fingerprint_detects_same_size_middle_edits() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("source.mp4");
        let renamed = dir.path().join("alias.mp4");
        let mut bytes = vec![0xAB; 3 * 65_536];
        std::fs::write(&source, &bytes).unwrap();
        std::fs::write(&renamed, &bytes).unwrap();
        let original = media_fingerprint(&source).unwrap();
        assert_eq!(original, media_fingerprint(&renamed).unwrap());
        assert_eq!(
            original,
            format!(
                "whole-pcm16-mono16k-strict-v3-{}",
                blake3::hash(&bytes).to_hex()
            )
        );
        bytes[65_537] = 1;
        std::fs::write(&source, &bytes).unwrap();
        assert_ne!(original, media_fingerprint(&source).unwrap());
    }

    /// Guards: Explicit cache location takes precedence over the platform directory.
    /// Kind: policy
    /// Not a type: Two valid path sources can disagree and still require a
    /// deliberate choice about precedence and the platform directory suffix.
    /// State: Two in-memory path sources with and without an override.
    /// Concurrency: No environment or files are modified.
    #[test]
    fn default_cache_dir_prefers_explicit_override() {
        let platform = PathBuf::from("/tmp/platform-data");
        let override_path = PathBuf::from("/tmp/media-cache");
        assert_eq!(
            default_cache_dir_from(Some(override_path.clone()), Some(platform.clone())),
            override_path
        );
        assert_eq!(
            default_cache_dir_from(None, Some(platform.clone())),
            platform.join("batchalign3").join(CACHE_SUBDIR)
        );
    }

    /// Guards: An absent source refuses segment preparation before the cache substitutes a result.
    /// Kind: contract
    /// Not a type: A valid bound and path can outlive the actual file they describe.
    /// State: An absent path in a temporary directory and one proven test window.
    /// Concurrency: The test owns the directory and awaits the real read failure.
    #[tokio::test]
    async fn segment_refuses_unreadable_source_identity() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("absent.wav");
        let recording = crate::chat_ops::fa::coordinates::Recording::of_duration(
            crate::chat_ops::fa::coordinates::Ms(1_000),
        )
        .unwrap();
        let window = FaWindow::within(
            &recording,
            crate::time::FileMs::new(0),
            crate::time::FileMs::new(100),
        )
        .unwrap();
        assert!(matches!(extract_audio_segment(&missing, window).await,
            Err(EnsureWavError::Io(error)) if error.kind() == std::io::ErrorKind::NotFound));
    }

    /// Guards: Native preparation carries its original header into measurement.
    /// Kind: contract
    /// Not a type: The native probe can be invoked twice against the same
    /// path unless the first observation survives preparation.
    /// State: A real PCM source is removed after successful container admission.
    /// Concurrency: This test controls the file and both transitions.
    #[tokio::test]
    async fn native_preparation_measures_the_header_it_already_observed() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("recording.wav");
        crate::media::probe::fixtures::write_pcm_wav(&source, 1);
        let prepared = MediaPreparation::inspect(source.clone()).await.unwrap();
        let MediaPreparation::Native(native) = prepared else {
            panic!("PCM must be admitted for passthrough");
        };
        std::fs::remove_file(&source).unwrap();
        let measured = native.measure().await;
        assert_eq!(measured.path(), source);
        let duration = measured.duration().unwrap();
        assert_eq!(duration.length().0, 100);
    }

    /// Guards: Actual containers select native passthrough or one shared PCM conversion,
    /// FA measures the prepared PCM timeline, and cache operations see only the published artifact.
    /// Kind: contract.
    /// Not a type: External container bytes and the real ffprobe/ffmpeg adapters
    /// determine whether a file can enter the admitted conversion path.
    /// State: One generated QuickTime/MP3 recording, renamed aliases and a
    /// remuxed native MPEG control and real content-addressed WAV cache exercise
    /// format detection, publication, accounting and clearing.
    /// Concurrency: A private temporary directory isolates files and cache entries;
    /// the production cache-slot owner serializes publication for identical content.
    #[tokio::test]
    async fn container_admission_converts_quicktime_under_native_suffixes() {
        use crate::media::probe::{DurationBasis, StatingContainer};

        let dir = tempfile::tempdir().expect("fixture directory");
        let source = dir.path().join("recording.mov");
        let encoded = MediaTool::Ffmpeg
            .async_command()
            .args([
                "-y",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:sample_rate=44100",
                "-t",
                "0.25",
                "-c:a",
                "libmp3lame",
                "-f",
                "mov",
            ])
            .arg(&source)
            .output()
            .await
            .expect("encode QuickTime fixture");
        assert!(
            encoded.status.success(),
            "{}",
            String::from_utf8_lossy(&encoded.stderr)
        );
        let original = std::fs::read(&source).expect("read encoded source");
        let cache = dir.path().join("cache");
        let mut published = std::collections::BTreeSet::new();
        for (index, name) in [
            "recording.m4a",
            "recording.mp4",
            "recording.mp3",
            "recording.wav",
            "recording",
        ]
        .into_iter()
        .enumerate()
        {
            let alias = dir.path().join(name);
            std::fs::write(&alias, &original).expect("write identical source alias");
            let prepared = if index == 0 {
                let measured = prepare_media(&alias, Some(&cache))
                    .await
                    .expect("prepare actual container")
                    .measure()
                    .await;
                assert_eq!(
                    measured.duration().expect("measure prepared WAV").basis(),
                    DurationBasis::Stated(StatingContainer::PcmWav)
                );
                measured.path().to_owned()
            } else {
                ensure_wav(&alias, Some(&cache))
                    .await
                    .expect("prepare identical source alias")
            };
            assert!(
                prepared.starts_with(&cache),
                "container requires decoded WAV for {name}"
            );
            assert_eq!(
                &std::fs::read(&prepared).expect("read PCM WAV")[..4],
                b"RIFF"
            );
            assert_eq!(
                std::fs::read(&alias).expect("source remains readable"),
                original
            );
            published.insert(prepared);
        }
        assert_eq!(published.len(), 1, "identical bytes share one conversion");

        // The opposite mismatch must also follow bytes: native MP3 inside a
        // filename normally associated with ISO media retains its own path.
        let native = dir.path().join("native.m4a");
        let remuxed = MediaTool::Ffmpeg
            .async_command()
            .args(["-y", "-i"])
            .arg(&source)
            .args(["-c:a", "copy", "-f", "mp3"])
            .arg(&native)
            .output()
            .await
            .expect("remux native MPEG control");
        assert!(
            remuxed.status.success(),
            "{}",
            String::from_utf8_lossy(&remuxed.stderr)
        );
        assert_eq!(
            ensure_wav(&native, Some(&cache))
                .await
                .expect("admit native MPEG container"),
            native
        );

        let (_, size, count) = cache_stats(Some(&cache));
        assert!(size > 0, "published WAV has bytes");
        assert_eq!(count, 1, "aliases share one published artifact");
        assert_eq!(clear_cache(Some(&cache)).expect("clear owned cache"), size);
        let (_, size, count) = cache_stats(Some(&cache));
        assert_eq!((size, count), (0, 0));
    }
}
