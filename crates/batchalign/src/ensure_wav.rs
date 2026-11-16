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

use thiserror::Error;
use tracing::{debug, info};

use crate::chat_ops::fa::coordinates::FaWindow;
use crate::media::probe::{MediaContainer, MediaProbe, ProbeError};
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
struct NativeContainerSource(PathBuf);

/// Only conversion admission can produce a source for the PCM cache.
struct ConversionSource(PathBuf);

impl MediaPreparation {
    async fn inspect(source: PathBuf) -> Result<Self, ProbeError> {
        match MediaProbe::new(&source).container().await? {
            MediaContainer::MpegAudio
            | MediaContainer::Wav
            | MediaContainer::Flac
            | MediaContainer::Ogg => Ok(Self::Native(NativeContainerSource(source))),
            MediaContainer::Adts
            | MediaContainer::IsoMedia
            | MediaContainer::Matroska
            | MediaContainer::Asf => Ok(Self::Convert(ConversionSource(source))),
        }
    }
}

impl NativeContainerSource {
    fn into_path(self) -> PathBuf {
        self.0
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
    fn produce(self, transcode: Transcode) -> Result<ProducedCacheSlot, TranscodeError> {
        transcode.produce(&self.tmp_path)?;
        Ok(ProducedCacheSlot { slot: self })
    }
}

/// A locked slot whose own temporary destination passed strict conversion.
struct ProducedCacheSlot {
    slot: LockedCacheSlot,
}

impl ProducedCacheSlot {
    /// Publish atomically before releasing the lock.
    fn commit(self) -> Result<PathBuf, std::io::Error> {
        let LockedCacheSlot {
            lock,
            tmp_path,
            cached_path,
        } = self.slot;
        std::fs::rename(&tmp_path, &cached_path)?;
        // Explicit, and AFTER the rename: this order is the entire point of the
        // type, so it is stated rather than left to end-of-scope drop order.
        drop(lock);
        Ok(cached_path)
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
    match MediaPreparation::inspect(source_path.to_path_buf()).await? {
        MediaPreparation::Native(source) => Ok(source.into_path()),
        MediaPreparation::Convert(source) => source.convert(cache_dir).await,
    }
}

impl ConversionSource {
    async fn convert(self, cache_dir: Option<&Path>) -> Result<PathBuf, EnsureWavError> {
        // Move all blocking I/O (fingerprint, file lock, ffmpeg subprocess,
        // rename) onto a dedicated thread so we don't starve the tokio executor.
        let source_path = self.0;
        let cache_dir = cache_dir.map(Path::to_path_buf);

        crate::blocking::spawn_in_span(move || -> Result<PathBuf, EnsureWavError> {
            let effective_cache_dir = match cache_dir {
                Some(p) => p,
                None => default_cache_dir(),
            };
            std::fs::create_dir_all(&effective_cache_dir)?;

            let fp = media_fingerprint(&source_path)?;
            let cached_wav = effective_cache_dir.join(format!("{fp}.wav"));

            // Fast path: already converted
            if cached_wav.exists() {
                debug!(
                    "Media cache hit: {} -> {}",
                    source_path.display(),
                    cached_wav.display()
                );
                return Ok(cached_wav);
            }

            // Convert under a per-fingerprint exclusive lock, held until the result
            // is published, which `LockedCacheSlot` is what guarantees.
            let slot = match LockedCacheSlot::acquire(&effective_cache_dir, &fp)? {
                CacheSlot::AlreadyPublished(path) => return Ok(path),
                CacheSlot::Ours(slot) => slot,
            };

            info!(
                "Converting {} -> {}",
                source_path.display(),
                cached_wav.display()
            );

            // The status check, the partial-file cleanup and the stderr reading all
            // live in `produce`; this call site keeps only what is its own.
            let produced = slot.produce(Transcode::whole(&source_path, PcmEncoding::S16LeWav))?;
            Ok(produced.commit()?)
        })
        .await
        .map_err(std::io::Error::other)?
    }
}

/// Extract a time-windowed audio segment to a cached WAV file.
///
/// Uses ffmpeg `-ss {start} -to {end}` with pcm_s16le 16kHz mono.
/// Result is cached in media_cache keyed by source fingerprint + time window.
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

        let produced = slot.produce(Transcode::window(&source, window, PcmEncoding::S16LeWav))?;
        Ok(produced.commit()?)
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

    #[test]
    fn fingerprint_deterministic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.mp4");
        std::fs::write(&path, b"fake mp4 content for fingerprinting").unwrap();

        let fp1 = media_fingerprint(&path).unwrap();
        let fp2 = media_fingerprint(&path).unwrap();
        assert_eq!(fp1, fp2, "fingerprint should be deterministic");
        assert_eq!(
            fp1,
            format!(
                "whole-pcm16-mono16k-strict-v3-{}",
                blake3::hash(b"fake mp4 content for fingerprinting").to_hex()
            )
        );
    }

    #[test]
    fn fingerprint_differs_for_different_content() {
        let dir = tempfile::tempdir().unwrap();
        let path_a = dir.path().join("a.mp4");
        let path_b = dir.path().join("b.mp4");
        std::fs::write(&path_a, b"content A").unwrap();
        std::fs::write(&path_b, b"content B").unwrap();

        let fp_a = media_fingerprint(&path_a).unwrap();
        let fp_b = media_fingerprint(&path_b).unwrap();
        assert_ne!(fp_a, fp_b);
    }

    #[test]
    fn fingerprint_detects_same_size_middle_edits() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("source.mp4");
        let mut bytes = vec![0u8; 3 * 65536];
        std::fs::write(&path, &bytes).unwrap();
        let before = media_fingerprint(&path).unwrap();
        bytes[65536 + 1] = 1;
        std::fs::write(&path, &bytes).unwrap();
        assert_ne!(before, media_fingerprint(&path).unwrap());
    }

    #[test]
    fn fingerprint_large_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("large.mp4");
        // Exercise streaming beyond the former sampled blocks.
        let data = vec![0xABu8; 65536 * 3];
        std::fs::write(&path, &data).unwrap();

        let fp = media_fingerprint(&path).unwrap();
        assert_eq!(
            fp,
            format!(
                "whole-pcm16-mono16k-strict-v3-{}",
                blake3::hash(&data).to_hex()
            )
        );
    }

    #[test]
    fn default_cache_dir_prefers_explicit_override() {
        let resolved = default_cache_dir_from(
            Some(PathBuf::from("/tmp/media-cache")),
            Some(PathBuf::from("/tmp/platform-data")),
        );

        assert_eq!(resolved, PathBuf::from("/tmp/media-cache"));
    }

    #[test]
    fn default_cache_dir_uses_platform_data_dir_when_no_override() {
        let resolved = default_cache_dir_from(None, Some(PathBuf::from("/tmp/platform-data")));

        assert_eq!(
            resolved,
            PathBuf::from("/tmp/platform-data")
                .join("batchalign3")
                .join(CACHE_SUBDIR)
        );
    }

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

    /// Guards: Actual containers select native passthrough or one shared PCM conversion,
    /// and cache accounting and clearing observe only the published artifact.
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
        for name in [
            "recording.m4a",
            "recording.mp4",
            "recording.mp3",
            "recording.wav",
            "recording",
        ] {
            let alias = dir.path().join(name);
            std::fs::write(&alias, &original).expect("write identical source alias");
            let prepared = ensure_wav(&alias, Some(&cache))
                .await
                .expect("prepare actual container");
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
