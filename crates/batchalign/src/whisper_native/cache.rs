//! Process-level cache for the loaded `WhisperContext`.
//!
//! `WhisperContext::new_with_params` reads a 1-3 GB ggml model from disk
//! and runs ~3 s of init on cold load. Production callers run hundreds of
//! transcribe jobs back-to-back; reloading per call would dominate wall
//! time. This module holds a single `Arc<WhisperContext>` per process,
//! lazy-initialized on first request, reused for the rest of the process
//! lifetime. `WhisperContext` is `Send + Sync`, so `Arc` cloning is the
//! natural sharing primitive.
//!
//! ## Single-slot policy
//!
//! The cache holds exactly one (path, context) pair per process. If a
//! second request asks for a different model path, this returns
//! `ModelPathChanged` rather than evicting and reloading; silently
//! reloading would (a) cost another 3 s + 3 GB, (b) leak the old context
//! until refcount drops, and (c) hide a misconfiguration where two
//! components disagree on which model to use. The fleet runs one model
//! per host (set via `BATCHALIGN_WHISPER_RS_MODEL`); a path change at
//! runtime is a bug, not a feature.
//!
//! ## Shutdown (why this is an `RwLock`, not a `OnceLock`)
//!
//! A `WhisperContext` owns Metal buffers. A Rust `static` never drops,
//! so a `OnceLock`-held context outlives ggml's own C++ static
//! destructors, which assert at process exit that every Metal resource
//! was deallocated (`ggml_metal_rsets_free`: "you haven't deallocated
//! all Metal resources before exiting", observed as a SIGABRT after a
//! fully successful transcription, 2026-07-29). `shutdown()` drops the
//! cached context before exit; the binary's epilogue calls it. The
//! read path stays cheap (uncontended `RwLock::read`), and the loader
//! runs OUTSIDE the lock, so two cold concurrent callers may both load
//! and the loser's copy is dropped (one wasted load, same trade the
//! previous `OnceLock` design made).

use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use super::error::WhisperNativeError;

/// Single-slot cache for a heavyweight loaded resource keyed by file
/// path. Generic over the loaded type so the production cache (holding
/// `Arc<WhisperContext>`) and unit tests (holding cheap stand-ins) share
/// the same logic.
///
/// `dead_code` allowance covers the no-feature build, where the
/// production consumer in `backend.rs` is feature-gated out but the
/// unit tests in this file (run with or without the feature) keep the
/// type exercised.
#[allow(dead_code)]
pub(super) struct PathKeyedCache<T> {
    cell: RwLock<Option<(PathBuf, Arc<T>)>>,
}

#[allow(dead_code)]
impl<T> PathKeyedCache<T> {
    pub(super) const fn new() -> Self {
        Self {
            cell: RwLock::new(None),
        }
    }

    /// Drop the cached value (if any), releasing the loaded resource
    /// while the process can still tear it down cleanly. Returns whether
    /// something was dropped. A poisoned lock is treated as "nothing to
    /// drop": at shutdown there is no better recovery than proceeding.
    pub(super) fn shutdown(&self) -> bool {
        match self.cell.write() {
            Ok(mut guard) => guard.take().is_some(),
            Err(_) => false,
        }
    }

    /// Return the cached `Arc<T>` for `requested_path`, loading it via
    /// `loader` on first call. Returns `ModelPathChanged` if the cache
    /// is already populated with a different path.
    pub(super) fn get_or_load<L>(
        &self,
        requested_path: &Path,
        loader: L,
    ) -> Result<Arc<T>, WhisperNativeError>
    where
        L: FnOnce(&Path) -> Result<Arc<T>, WhisperNativeError>,
    {
        if let Some(hit) = Self::lookup(
            self.cell.read().map_err(Self::poisoned)?.as_ref(),
            requested_path,
        )? {
            return Ok(hit);
        }
        // Load OUTSIDE the lock: the 3 s / 3 GB model load must not hold
        // the cache closed.
        let value = loader(requested_path)?;
        let mut guard = self.cell.write().map_err(Self::poisoned)?;
        match Self::lookup(guard.as_ref(), requested_path)? {
            // A concurrent caller won the race; ours drops.
            Some(winner) => Ok(winner),
            None => {
                *guard = Some((requested_path.to_path_buf(), Arc::clone(&value)));
                Ok(value)
            }
        }
    }

    /// Shared read-side check: cache hit, path conflict, or empty.
    fn lookup(
        slot: Option<&(PathBuf, Arc<T>)>,
        requested_path: &Path,
    ) -> Result<Option<Arc<T>>, WhisperNativeError> {
        match slot {
            Some((cached_path, value)) if cached_path == requested_path => {
                Ok(Some(Arc::clone(value)))
            }
            Some((cached_path, _)) => Err(WhisperNativeError::ModelPathChanged {
                cached: cached_path.clone(),
                requested: requested_path.to_path_buf(),
            }),
            None => Ok(None),
        }
    }

    fn poisoned<E>(_: E) -> WhisperNativeError {
        WhisperNativeError::CacheInvariant("cache lock poisoned".to_string())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn first_call_invokes_loader() {
        let cache: PathKeyedCache<u32> = PathKeyedCache::new();
        let calls = AtomicUsize::new(0);
        let v = cache
            .get_or_load(Path::new("/m.bin"), |_| {
                calls.fetch_add(1, Ordering::SeqCst);
                Ok(Arc::new(42u32))
            })
            .unwrap();
        assert_eq!(*v, 42);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn second_call_same_path_reuses_arc() {
        let cache: PathKeyedCache<u32> = PathKeyedCache::new();
        let calls = AtomicUsize::new(0);
        let load = |_: &Path| {
            calls.fetch_add(1, Ordering::SeqCst);
            Ok(Arc::new(7u32))
        };
        let a = cache.get_or_load(Path::new("/m.bin"), load).unwrap();
        let b = cache.get_or_load(Path::new("/m.bin"), load).unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert!(Arc::ptr_eq(&a, &b));
    }

    #[test]
    fn different_path_returns_model_path_changed() {
        let cache: PathKeyedCache<u32> = PathKeyedCache::new();
        let _ = cache
            .get_or_load(Path::new("/first.bin"), |_| Ok(Arc::new(1u32)))
            .unwrap();
        let err = cache
            .get_or_load(Path::new("/second.bin"), |_| Ok(Arc::new(2u32)))
            .unwrap_err();
        match err {
            WhisperNativeError::ModelPathChanged { cached, requested } => {
                assert_eq!(cached, PathBuf::from("/first.bin"));
                assert_eq!(requested, PathBuf::from("/second.bin"));
            }
            other => panic!("expected ModelPathChanged, got {other:?}"),
        }
    }

    #[test]
    fn loader_error_is_propagated_and_cache_remains_uninitialized() {
        let cache: PathKeyedCache<u32> = PathKeyedCache::new();
        let err = cache
            .get_or_load(Path::new("/m.bin"), |_| {
                Err(WhisperNativeError::ModelPathMissing)
            })
            .unwrap_err();
        assert!(matches!(err, WhisperNativeError::ModelPathMissing));
        // Subsequent successful load still works; the cache wasn't
        // poisoned by the failed first attempt.
        let v = cache
            .get_or_load(Path::new("/m.bin"), |_| Ok(Arc::new(99u32)))
            .unwrap();
        assert_eq!(*v, 99);
    }

    #[test]
    fn shutdown_drops_the_cached_value_and_allows_reload() {
        let cache: PathKeyedCache<u32> = PathKeyedCache::new();
        let calls = AtomicUsize::new(0);
        let load = |_: &Path| {
            calls.fetch_add(1, Ordering::SeqCst);
            Ok(Arc::new(9u32))
        };
        let a = cache.get_or_load(Path::new("/m.bin"), load).unwrap();
        assert_eq!(Arc::strong_count(&a), 2);
        assert!(cache.shutdown());
        // The cache's reference is gone; ours is the only one left.
        assert_eq!(Arc::strong_count(&a), 1);
        // Nothing cached: shutdown again is a no-op...
        assert!(!cache.shutdown());
        // ...and a reload (even under a NEW path) is permitted.
        let b = cache.get_or_load(Path::new("/other.bin"), load).unwrap();
        assert_eq!(*b, 9);
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }
}
