//! Content-fingerprinted WAV conversion cache.
//!
//! Converts non-WAV media (e.g. mp4) to WAV via ffmpeg and caches the result
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
//! - **Locking**: per-fingerprint `.lock` file via `fs2` exclusive lock prevents
//!   concurrent ffmpeg invocations for the same source file (important for
//!   parallel FA groups).
//! - **Atomic writes**: convert to a temp file, then `rename()` into the cache.
//!
//! # Formats requiring conversion
//!
//! Only container formats that `libsndfile` (via `soundfile` in Python) cannot
//! read need conversion. Currently: `.mp4`, `.m4a`, `.webm`, `.wma`.
//! WAV, MP3, FLAC, OGG are handled natively by `soundfile`.

use std::path::{Path, PathBuf};

use fs2::FileExt;
use thiserror::Error;
use tracing::{debug, info};

use crate::media::transcode::{PcmEncoding, Transcode, TranscodeError};
use crate::media::window::MediaWindow;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Extensions that `soundfile` (libsndfile) cannot read and must be converted.
const FORCED_CONVERSION: &[&str] = &["mp4", "m4a", "webm", "wma"];

/// Default cache directory name under the app data dir.
const CACHE_SUBDIR: &str = "media_cache";

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
/// Errors from [`ensure_wav`].
pub enum EnsureWavError {
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
/// It used to be exactly that, and the comment above it said the opposite: "the
/// lock must be held across the conversion", four lines before
/// `lock_file.unlock()` and eight before the `rename` that actually publishes.
/// The window was real. Two callers converting the same fingerprint share one
/// `{key}.tmp.wav`: A unlocks, B takes the lock, B's re-check sees no cached
/// file because A has not renamed yet, B begins writing that same temp path,
/// and A renames B's half-written file into the cache. The fingerprinted cache
/// then serves it forever, because the fast path never looks inside.
///
/// Predates the `Transcode` work; present at `origin/main`.
struct LockedCacheSlot {
    /// Dropping this releases the lock, which is why `commit` renames first.
    lock_file: std::fs::File,
    tmp_path: PathBuf,
    cached_path: PathBuf,
}

impl LockedCacheSlot {
    /// Take the lock for `key`, or report that someone else got there first.
    fn acquire(cache_dir: &Path, key: &str) -> Result<CacheSlot, std::io::Error> {
        let cached_path = cache_dir.join(format!("{key}.wav"));
        let lock_file = std::fs::File::create(cache_dir.join(format!("{key}.wav.lock")))?;
        lock_file.lock_exclusive()?;
        // Re-check under the lock: another writer may have finished while we
        // waited. Now that publishing happens under the lock too, this answer
        // cannot go stale between the check and the caller acting on it.
        if cached_path.exists() {
            return Ok(CacheSlot::AlreadyPublished(cached_path));
        }
        Ok(CacheSlot::Ours(Self {
            lock_file,
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
            lock_file,
            tmp_path,
            cached_path,
        } = self.slot;
        std::fs::rename(&tmp_path, &cached_path)?;
        // Explicit, and AFTER the rename: this order is the entire point of the
        // type, so it is stated rather than left to end-of-scope drop order.
        drop(lock_file);
        Ok(cached_path)
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Return a path to a WAV version of `source_path`.
///
/// - If `source_path` is already `.wav` / `.mp3` / `.flac` / `.ogg` (formats
///   that `soundfile` handles natively), return it unchanged.
/// - If a cached conversion exists (by content fingerprint), return that.
/// - Otherwise convert via ffmpeg, cache the result, and return the cache path.
///
/// `cache_dir` overrides the default cache location (for testing).
pub async fn ensure_wav(
    source_path: &Path,
    cache_dir: Option<&Path>,
) -> Result<PathBuf, EnsureWavError> {
    if !needs_conversion(source_path) {
        return Ok(source_path.to_path_buf());
    }

    // Move all blocking I/O (fingerprint, fs2 lock, ffmpeg subprocess,
    // rename) onto a dedicated thread so we don't starve the tokio executor.
    let source_path = source_path.to_path_buf();
    let cache_dir = cache_dir.map(Path::to_path_buf);

    tokio::task::spawn_blocking(move || -> Result<PathBuf, EnsureWavError> {
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

/// Extract a time-windowed audio segment to a cached WAV file.
///
/// Uses ffmpeg `-ss {start} -to {end}` with pcm_s16le 16kHz mono.
/// Result is cached in media_cache keyed by source fingerprint + time window.
pub async fn extract_audio_segment(
    source: &Path,
    window: MediaWindow,
) -> Result<PathBuf, EnsureWavError> {
    let source = source.to_path_buf();
    // Only for the cache key: the window itself is already proven, so nothing
    // here re-checks it. That check used to live in this function, and in two
    // others, none of them where the pair originates.
    let (start_ms, end_ms) = (window.start().get(), window.end().get());

    tokio::task::spawn_blocking(move || -> Result<PathBuf, EnsureWavError> {
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

/// Check whether the file extension requires ffmpeg conversion.
pub fn needs_conversion(path: &Path) -> bool {
    let ext = path
        .extension()
        .map(|e| e.to_string_lossy().to_lowercase())
        .unwrap_or_default();
    FORCED_CONVERSION.contains(&ext.as_str())
}

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
        dirs::data_dir(),
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
    fn wav_needs_no_conversion() {
        assert!(!needs_conversion(Path::new("audio.wav")));
        assert!(!needs_conversion(Path::new("audio.WAV")));
        assert!(!needs_conversion(Path::new("/path/to/audio.mp3")));
        assert!(!needs_conversion(Path::new("audio.flac")));
        assert!(!needs_conversion(Path::new("audio.ogg")));
    }

    #[test]
    fn mp4_needs_conversion() {
        assert!(needs_conversion(Path::new("video.mp4")));
        assert!(needs_conversion(Path::new("video.MP4")));
        assert!(needs_conversion(Path::new("/path/to/video.mp4")));
    }

    #[test]
    fn m4a_needs_conversion() {
        assert!(needs_conversion(Path::new("audio.m4a")));
    }

    #[test]
    fn webm_needs_conversion() {
        assert!(needs_conversion(Path::new("video.webm")));
    }

    #[test]
    fn wma_needs_conversion() {
        assert!(needs_conversion(Path::new("audio.wma")));
    }

    #[test]
    fn no_extension_needs_no_conversion() {
        assert!(!needs_conversion(Path::new("noext")));
        assert!(!needs_conversion(Path::new("/path/to/noext")));
    }

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
        let window =
            MediaWindow::new(crate::time::FileMs::new(0), crate::time::FileMs::new(100)).unwrap();
        assert!(matches!(extract_audio_segment(&missing, window).await,
            Err(EnsureWavError::Io(error)) if error.kind() == std::io::ErrorKind::NotFound));
    }

    #[tokio::test]
    async fn ensure_wav_passthrough_for_wav() {
        let result = ensure_wav(Path::new("/some/path/audio.wav"), None)
            .await
            .unwrap();
        assert_eq!(result, Path::new("/some/path/audio.wav"));
    }

    #[tokio::test]
    async fn ensure_wav_passthrough_for_mp3() {
        let result = ensure_wav(Path::new("/some/path/audio.mp3"), None)
            .await
            .unwrap();
        assert_eq!(result, Path::new("/some/path/audio.mp3"));
    }

    #[tokio::test]
    async fn ensure_wav_passthrough_for_flac() {
        let result = ensure_wav(Path::new("/some/path/audio.flac"), None)
            .await
            .unwrap();
        assert_eq!(result, Path::new("/some/path/audio.flac"));
    }

    #[tokio::test]
    async fn ensure_wav_converts_mp4() {
        if MediaTool::Ffmpeg.banner().is_none() {
            eprintln!("skipping: ffmpeg not installed");
            return;
        }

        let dir = tempfile::tempdir().unwrap();
        let cache_dir = dir.path().join("cache");

        // Create a minimal valid mp4 via ffmpeg (short silent audio)
        let mp4_path = dir.path().join("test.mp4");
        let ffmpeg_out = MediaTool::Ffmpeg
            .async_command()
            .args([
                "-y",
                "-f",
                "lavfi",
                "-i",
                "anullsrc=r=16000:cl=mono",
                "-t",
                "0.1",
                mp4_path.to_string_lossy().as_ref(),
            ])
            .output()
            .await;
        if ffmpeg_out.is_err() || !ffmpeg_out.unwrap().status.success() {
            eprintln!("skipping: could not generate test mp4");
            return;
        }

        // First call should convert
        let wav_path = ensure_wav(&mp4_path, Some(&cache_dir)).await.unwrap();
        assert!(
            wav_path.extension().is_some_and(|e| e == "wav"),
            "should return a .wav path"
        );
        assert!(wav_path.exists(), "converted wav should exist on disk");
        assert!(
            wav_path.starts_with(&cache_dir),
            "wav should be in cache dir"
        );

        // Second call should hit cache
        let wav_path2 = ensure_wav(&mp4_path, Some(&cache_dir)).await.unwrap();
        assert_eq!(wav_path, wav_path2, "cache should return same path");
    }

    #[tokio::test]
    async fn ensure_wav_converts_m4a() {
        if MediaTool::Ffmpeg.banner().is_none() {
            eprintln!("skipping: ffmpeg not installed");
            return;
        }

        let dir = tempfile::tempdir().unwrap();
        let cache_dir = dir.path().join("cache");

        let m4a_path = dir.path().join("test.m4a");
        let ffmpeg_out = MediaTool::Ffmpeg
            .async_command()
            .args([
                "-y",
                "-f",
                "lavfi",
                "-i",
                "anullsrc=r=16000:cl=mono",
                "-t",
                "0.1",
                m4a_path.to_string_lossy().as_ref(),
            ])
            .output()
            .await;
        if ffmpeg_out.is_err() || !ffmpeg_out.unwrap().status.success() {
            eprintln!("skipping: could not generate test m4a");
            return;
        }

        let wav_path = ensure_wav(&m4a_path, Some(&cache_dir)).await.unwrap();
        assert!(wav_path.extension().is_some_and(|e| e == "wav"));
        assert!(wav_path.exists());
    }

    #[tokio::test]
    async fn ensure_wav_cache_stats_and_clear() {
        if MediaTool::Ffmpeg.banner().is_none() {
            eprintln!("skipping: ffmpeg not installed");
            return;
        }

        let dir = tempfile::tempdir().unwrap();
        let cache_dir = dir.path().join("cache");

        // Empty cache
        let (_, size, count) = cache_stats(Some(&cache_dir));
        assert_eq!(size, 0);
        assert_eq!(count, 0);

        // Generate and convert an mp4
        let mp4_path = dir.path().join("test.mp4");
        let ffmpeg_out = MediaTool::Ffmpeg
            .async_command()
            .args([
                "-y",
                "-f",
                "lavfi",
                "-i",
                "anullsrc=r=16000:cl=mono",
                "-t",
                "0.1",
                mp4_path.to_string_lossy().as_ref(),
            ])
            .output()
            .await;
        if ffmpeg_out.is_err() || !ffmpeg_out.unwrap().status.success() {
            return;
        }

        let _ = ensure_wav(&mp4_path, Some(&cache_dir)).await.unwrap();

        let (_, size, count) = cache_stats(Some(&cache_dir));
        assert!(size > 0, "cache should have nonzero size");
        assert_eq!(count, 1, "cache should have one file");

        // Clear
        let freed = clear_cache(Some(&cache_dir)).unwrap();
        assert!(freed > 0);
        let (_, size, count) = cache_stats(Some(&cache_dir));
        assert_eq!(size, 0);
        assert_eq!(count, 0);
    }
}
