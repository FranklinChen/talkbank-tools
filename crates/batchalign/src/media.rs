//! Media listing and lookup: originally a port of `batchalign/serve/media.py`.
//!
//! Lists configured media_roots for audio/video files. Results are cached
//! with a 60-second TTL to avoid rescanning NFS mounts on every request.
//!
//! Finding media and PROCESSING it are both media concerns, so the external
//! tools live below this module as [`tools`]. They are otherwise unrelated:
//! nothing here spawns anything.

pub(crate) mod access;
pub mod declared;
pub mod export;
pub mod extensions;
pub mod probe;
pub mod tools;
pub mod transcode;
pub mod window;

use std::path::{Path, PathBuf};
use std::time::Instant;

pub use declared::DeclaredMedia;
use extensions::DirWalk;
pub use extensions::{MediaExtensions, MediaLookup, Missed};

use dashmap::DashMap;
use tracing::{debug, warn};

/// Walk cache TTL in seconds.
///
/// Media roots are often NFS-mounted volumes with tens of thousands of files.
/// Caching walk results for 60 seconds avoids re-scanning on every request
/// while still picking up new files within a reasonable window. The cache is
/// per-root, so adding a new root does not invalidate existing entries.
const CACHE_TTL_SECS: u64 = 60;

/// One discovered media file: the directory it lives in and its filename.
#[derive(Debug, Clone)]
pub struct MediaEntry {
    /// Parent directory path (e.g. `/data/media/subdir`).
    pub dir_path: String,
    /// Filename with extension (e.g. `interview.wav`).
    pub filename: String,
}

impl MediaEntry {
    /// Full path to the media file.
    ///
    /// Uses `Path::join` for platform-safe path construction (correct on
    /// Windows where the separator is `\`, not `/`).
    pub fn full_path(&self) -> String {
        std::path::Path::new(&self.dir_path)
            .join(&self.filename)
            .to_string_lossy()
            .into_owned()
    }
}

/// `root/subdir` reached by exact names, or `None` if it is not there as
/// spelled.
///
/// Names are checked BEFORE `canonicalize`, which on macOS returns the on-disk
/// spelling and so would list `Brown/` for a request for `brown/` there and
/// nothing on a case-sensitive host (see [`MediaLookup`]). A subdir spelled
/// differently (letter case or Unicode form) is logged and lists nothing, the
/// same on every host. The
/// callers' `canonicalize` then still guards against a symlink leading out of
/// the root, which a name check cannot see.
fn exact_subdir(root: &str, subdir: &str) -> Option<PathBuf> {
    match DirWalk::of(Path::new(root), Path::new(subdir)) {
        Ok(walk) if walk.is_exact() => Some(walk.on_disk),
        Ok(walk) => {
            warn!(
                wanted = %walk.wanted.display(),
                on_disk = %walk.on_disk.display(),
                "Media subdir is spelled differently (letter case or Unicode form); nothing listed"
            );
            None
        }
        Err(_) => None,
    }
}

/// Cached walk results: `(timestamp, entries)`.
type CacheEntry = (Instant, Vec<MediaEntry>);

/// Lists audio/video files across configured media roots and named media
/// mappings, for `/media/list`.
///
/// Two listings:
/// - **Root listing** (`list_files`): walks all `media_roots` recursively.
/// - **Mapped listing** (`list_mapped`): restricted to one
///   `mapping_root/subdir`, with traversal protection.
///
/// It resolves nothing. Finding the recording for a transcript is
/// `runner::dispatch::media_search`, through
/// [`MediaExtensions`]'s exact-name lookup. This type's own `resolve` and
/// `resolve_mapped` had no production caller and were deleted on 2026-09-29;
/// the mapped one `canonicalize`d its directory, which on macOS silently
/// repairs a directory spelled differently (see [`MediaLookup`]).
///
/// Walk results are cached in a concurrent `DashMap` with a 60-second TTL
/// to avoid rescanning large NFS volumes on every request. The cache can be
/// invalidated per-root or globally via [`invalidate`](Self::invalidate).
#[derive(Clone)]
pub struct MediaResolver {
    cache: std::sync::Arc<DashMap<String, CacheEntry>>,
}

impl MediaResolver {
    /// Create a resolver with an empty walk cache.
    pub fn new() -> Self {
        Self {
            cache: std::sync::Arc::new(DashMap::new()),
        }
    }

    /// List a selected mapping with bounded filesystem access.
    pub(crate) async fn list_mapped_bounded(
        &self,
        root: String,
        subdir: String,
    ) -> Result<Vec<String>, access::MediaAccessError> {
        let resolver = self.clone();
        match access::access(root.into(), move |root| {
            resolver.list_mapped(&root.path().to_string_lossy(), &subdir)
        })
        .await
        {
            Err(access::MediaAccessError::Missing { .. }) => Ok(Vec::new()),
            result => result,
        }
    }

    /// List configured roots on demand, failing explicitly on an unresponsive root.
    pub(crate) async fn list_files_bounded(
        &self,
        roots: Vec<String>,
        subdir: String,
    ) -> Result<Vec<String>, access::MediaAccessError> {
        let mut files = Vec::new();
        for root in roots {
            let resolver = self.clone();
            let subdir = subdir.clone();
            match access::access(root.into(), move |root| {
                resolver.list_files(&[root.path().to_string_lossy().into_owned()], &subdir)
            })
            .await
            {
                Ok(found) => files.extend(found),
                Err(access::MediaAccessError::Missing { .. }) => {}
                Err(error) => return Err(error),
            }
        }
        files.sort();
        files.dedup();
        Ok(files)
    }

    /// Invalidate a specific root or the entire cache.
    pub fn invalidate(&self, root_dir: Option<&str>) {
        match root_dir {
            Some(root) => {
                self.cache.remove(root);
            }
            None => {
                self.cache.clear();
            }
        }
    }

    /// Walk a directory tree and return discovered media entries.
    ///
    /// Results are cached for `CACHE_TTL_SECS` seconds.
    fn walk_media(&self, root_dir: &str) -> Vec<MediaEntry> {
        let now = Instant::now();

        // Check cache
        if let Some(entry) = self.cache.get(root_dir) {
            let (ts, ref cached) = *entry;
            if now.duration_since(ts).as_secs() < CACHE_TTL_SECS {
                return cached.clone();
            }
        }

        // Cache miss: do the walk
        let mut entries = Vec::new();
        let root_path = Path::new(root_dir);
        if root_path.is_dir() {
            for entry in walkdir::WalkDir::new(root_path) {
                let entry = match entry {
                    Ok(entry) => entry,
                    Err(error) => {
                        warn!(
                            root = %root_dir,
                            error = %error,
                            "Skipping unreadable media walk entry"
                        );
                        continue;
                    }
                };
                if entry.file_type().is_file() {
                    let filename = entry.file_name().to_string_lossy().to_string();
                    if MediaExtensions::matches(&filename) {
                        let Some(parent) = entry.path().parent() else {
                            warn!(
                                path = %entry.path().display(),
                                "Skipping media file without parent directory"
                            );
                            continue;
                        };
                        let dir_path = parent.to_string_lossy().to_string();
                        entries.push(MediaEntry { dir_path, filename });
                    }
                }
            }
        } else if root_path.exists() {
            warn!(root = %root_dir, "Configured media root is not a directory");
        } else {
            debug!(root = %root_dir, "Configured media root does not exist");
        }

        self.cache
            .insert(root_dir.to_string(), (now, entries.clone()));
        entries
    }

    /// List audio/video filenames under a mapping root + subdir.
    fn list_mapped(&self, mapping_root: &str, subdir: &str) -> Vec<String> {
        let Some(search_dir) = exact_subdir(mapping_root, subdir) else {
            return Vec::new();
        };
        let search_dir = match search_dir.canonicalize() {
            Ok(p) => p,
            Err(_) => return Vec::new(),
        };
        let root_resolved = match PathBuf::from(mapping_root).canonicalize() {
            Ok(p) => p,
            Err(_) => return Vec::new(),
        };

        if !search_dir.starts_with(&root_resolved) {
            return Vec::new(); // path traversal
        }
        if !search_dir.is_dir() {
            return Vec::new();
        }

        let entries = self.walk_media(&search_dir.to_string_lossy());
        let mut names: Vec<String> = entries.into_iter().map(|e| e.filename).collect();
        names.sort();
        names.dedup();
        names
    }

    /// List audio/video filenames available under media_roots.
    fn list_files(&self, media_roots: &[String], subdir: &str) -> Vec<String> {
        let mut found = Vec::new();

        for root in media_roots {
            let search_dir = if subdir.is_empty() {
                root.clone()
            } else {
                let Some(search_path) = exact_subdir(root, subdir) else {
                    continue;
                };
                let search_resolved = match search_path.canonicalize() {
                    Ok(p) => p,
                    Err(_) => continue,
                };
                let root_resolved = match PathBuf::from(root).canonicalize() {
                    Ok(p) => p,
                    Err(_) => continue,
                };
                if !search_resolved.starts_with(&root_resolved) {
                    continue; // path traversal
                }
                search_resolved.to_string_lossy().to_string()
            };

            let entries = self.walk_media(&search_dir);
            for entry in entries {
                found.push(entry.filename);
            }
        }

        found.sort();
        found.dedup();
        found
    }
}

impl Default for MediaResolver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn setup_media_dir() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        fs::write(root.join("audio.wav"), b"fake wav").unwrap();
        fs::write(root.join("video.mp4"), b"fake mp4").unwrap();
        fs::write(root.join("song.mp3"), b"fake mp3").unwrap();
        fs::write(root.join("notes.txt"), b"not media").unwrap();

        let sub = root.join("subdir");
        fs::create_dir(&sub).unwrap();
        fs::write(sub.join("deep.flac"), b"fake flac").unwrap();
        dir
    }

    #[tokio::test]
    async fn bounded_listing_uses_the_mapping_and_walk_cache() {
        let dir = setup_media_dir();
        let resolver = MediaResolver::new();
        let root = dir.path().to_string_lossy().into_owned();
        assert_eq!(
            resolver
                .list_mapped_bounded(root.clone(), "subdir".into())
                .await
                .unwrap(),
            vec!["deep.flac"]
        );
        let files = resolver
            .list_files_bounded(vec![root], String::new())
            .await
            .unwrap();
        assert_eq!(
            files,
            vec!["audio.wav", "deep.flac", "song.mp3", "video.mp4"]
        );
    }

    #[test]
    fn list_files_all() {
        let dir = setup_media_dir();
        let resolver = MediaResolver::new();
        let roots = vec![dir.path().to_string_lossy().to_string()];

        let files = resolver.list_files(&roots, "");
        assert!(files.contains(&"audio.wav".to_string()));
        assert!(files.contains(&"video.mp4".to_string()));
        assert!(files.contains(&"deep.flac".to_string()));
        assert!(!files.contains(&"notes.txt".to_string()));
    }

    #[test]
    fn cache_invalidation() {
        let dir = setup_media_dir();
        let resolver = MediaResolver::new();
        let root = dir.path().to_string_lossy().to_string();
        let roots = vec![root.clone()];

        // Populate cache
        let _ = resolver.list_files(&roots, "");
        assert!(!resolver.cache.is_empty());

        // Invalidate
        resolver.invalidate(Some(&root));
        assert!(resolver.cache.is_empty());
    }

    #[test]
    fn list_mapped() {
        let dir = setup_media_dir();
        let resolver = MediaResolver::new();
        let root = dir.path().to_string_lossy().to_string();

        let files = resolver.list_mapped(&root, "subdir");
        assert_eq!(files, vec!["deep.flac"]);
    }

    #[test]
    fn cache_hit_avoids_rewalk() {
        let dir = setup_media_dir();
        let resolver = MediaResolver::new();
        let root = dir.path().to_string_lossy().to_string();
        let roots = vec![root.clone()];

        // First call populates the cache.
        let files1 = resolver.list_files(&roots, "");
        assert!(!files1.is_empty());

        // Add a new file to disk.
        fs::write(dir.path().join("new.wav"), b"new wav").unwrap();

        // Second call should use cached results (within TTL) and NOT see the new file.
        let files2 = resolver.list_files(&roots, "");
        assert_eq!(files1, files2, "cache hit should return same results");
        assert!(
            !files2.contains(&"new.wav".to_string()),
            "new file should not appear until cache expires"
        );
    }

    #[test]
    fn invalidation_reveals_new_files() {
        let dir = setup_media_dir();
        let resolver = MediaResolver::new();
        let root = dir.path().to_string_lossy().to_string();
        let roots = vec![root.clone()];

        // Populate cache.
        let files1 = resolver.list_files(&roots, "");
        assert!(!files1.contains(&"new.wav".to_string()));

        // Add a file and invalidate.
        fs::write(dir.path().join("new.wav"), b"new wav").unwrap();
        resolver.invalidate(Some(&root));

        // Now the new file should appear.
        let files2 = resolver.list_files(&roots, "");
        assert!(
            files2.contains(&"new.wav".to_string()),
            "after invalidation, new file should be discovered"
        );
    }

    #[test]
    fn global_invalidation_clears_all_roots() {
        let dir1 = setup_media_dir();
        let dir2 = tempfile::tempdir().unwrap();
        fs::write(dir2.path().join("track.mp3"), b"mp3").unwrap();

        let resolver = MediaResolver::new();
        let root1 = dir1.path().to_string_lossy().to_string();
        let root2 = dir2.path().to_string_lossy().to_string();
        let roots = vec![root1, root2];

        // Populate cache for both roots.
        let _ = resolver.list_files(&roots, "");
        assert!(resolver.cache.len() >= 2, "both roots should be cached");

        // Global invalidation.
        resolver.invalidate(None);
        assert!(resolver.cache.is_empty(), "all entries should be cleared");
    }

    #[test]
    fn nonexistent_root_returns_empty() {
        let resolver = MediaResolver::new();
        let roots = vec!["/nonexistent/path/that/does/not/exist".to_string()];

        let files = resolver.list_files(&roots, "");
        assert!(files.is_empty(), "non-existent root should yield no files");
    }

    #[test]
    fn non_media_files_excluded() {
        let dir = setup_media_dir();
        let resolver = MediaResolver::new();
        let roots = vec![dir.path().to_string_lossy().to_string()];

        let files = resolver.list_files(&roots, "");
        assert!(
            !files.contains(&"notes.txt".to_string()),
            ".txt should not appear in media listing"
        );
        assert!(
            files.contains(&"audio.wav".to_string()),
            ".wav should appear"
        );
        assert!(
            files.contains(&"video.mp4".to_string()),
            ".mp4 should appear"
        );
    }

    /// On macOS, `canonicalize` alone would list `subdir/` for `SUBDIR`; the
    /// answer must be the same on every host.
    #[test]
    fn a_subdir_spelled_with_other_case_lists_nothing() {
        let dir = setup_media_dir();
        let resolver = MediaResolver::new();
        let root = dir.path().to_string_lossy().to_string();
        assert_eq!(resolver.list_mapped(&root, "subdir"), vec!["deep.flac"]);
        assert!(resolver.list_mapped(&root, "SUBDIR").is_empty());
        assert!(
            !resolver
                .list_files(std::slice::from_ref(&root), "SUBDIR")
                .contains(&"deep.flac".to_owned())
        );
    }

    #[test]
    fn mapped_traversal_blocked() {
        let dir = setup_media_dir();
        let resolver = MediaResolver::new();
        let root = dir.path().to_string_lossy().to_string();

        // A subdir that climbs out of the mapping root lists nothing.
        let files = resolver.list_mapped(&root, "../../../etc");
        assert!(files.is_empty(), "path traversal should be blocked");
    }
}
