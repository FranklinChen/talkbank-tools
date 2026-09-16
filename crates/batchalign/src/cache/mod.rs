//! Durable audio-evidence cache for the batchalign3 server.
//!
//! Text-NLP caching was removed after production benchmarks showed it cost
//! more than warm inference. This module persists expensive audio results:
//! forced alignment, UTR ASR, raw Rev transcripts, and dedicated speaker
//! evidence. Each entry uses a task-specific BLAKE3 key and exact task/model
//! scope. Rev and dedicated-speaker paid boundaries additionally carry a typed
//! miss authorization and a process-local [`InferenceLease`] so concurrent
//! identical work is single-flight.
//!
//! # Python compatibility
//!
//! The SQLite schema, key formulas, and database file path are identical to
//! the Python `CacheManager` in `batchalign/pipelines/cache.py`.  Both Rust
//! and Python processes can read and write the same `cache.db` concurrently
//! thanks to SQLite WAL mode.  This allows the Rust server and the Python
//! processing pipeline to share a single cache during the migration period.
//!
//! # Database location
//!
//! The default database path follows platform conventions via the [`dirs`]
//! crate, matching Python's `platformdirs.user_cache_dir("batchalign3",
//! "batchalign3")`:
//!
//! | Platform | Path |
//! |----------|------|
//! | macOS    | `~/Library/Caches/batchalign3/cache.db` |
//! | Linux    | `~/.cache/batchalign3/cache.db` |
//!
//! A custom directory can be passed to [`UtteranceCache::sqlite`] for test
//! isolation.
//!
//! # Architecture
//!
//! The crate is organized around the [`CacheBackend`] trait, which defines
//! the storage contract (get, put, delete -- both single and batched).
//!
//! The production configuration is a **tiered cache**: a
//! [`TieredCacheBackend`] wrapping [`SqliteBackend`].  The hot layer is a
//! [moka](https://github.com/moka-rs/moka) `future::Cache` (10,000 entries,
//! 24h time-to-idle) that absorbs repeated lookups and reduces SQLite
//! round-trips under concurrent workloads.  SQLite remains the authoritative
//! persistent store; writes go through both layers (write-through).
//!
//! [`UtteranceCache`] is the public entry point.  It wraps a
//! `Box<dyn CacheBackend>` and provides factory methods:
//!
//! - [`UtteranceCache::tiered`] -- open a tiered cache (moka hot + SQLite
//!   cold).  This is the default used in production.
//! - [`UtteranceCache::sqlite`] -- open a plain SQLite cache (no hot layer).
//! - [`UtteranceCache::from_backend`] -- inject a custom backend (e.g. for
//!   testing with an in-memory store).
//!
//! # Typed tasks and namespaces
//!
//! Every row is scoped by a task name and a namespace (the model identity the
//! row was produced under; a different namespace is a miss). A backend stores
//! both as strings, but `UtteranceCache` does not accept strings: each cache
//! task is a [`CacheTask`] constant in [`tasks`] that fixes the namespace TYPE
//! its rows use, and every read and write takes that constant with a value of
//! that type. Handing the FA engine to the UTR task, or a speaker revision to
//! the Rev task, does not compile. The namespace types implement the sealed
//! [`CacheNamespace`] trait, and the task constants can only be made here, so
//! there is no route to a mismatched pair.
//!
//! `UtteranceCache` deliberately does not implement [`CacheBackend`] itself:
//! that trait takes bare strings, and exposing it on the typed wrapper would be
//! a route around the pairing.
//!
//! # Modules
//!
//! | Module      | Purpose |
//! |-------------|---------|
//! | [`backend`] | [`CacheBackend`] trait definition and [`CacheStats`] type |
//! | `sqlite`    | [`SqliteBackend`] -- WAL-mode SQLite implementation |
//! | `tiered`    | [`TieredCacheBackend`] -- moka hot layer + cold backend |
//!
//! # Examples
//!
//! The string-keyed storage contract, on a backend directly:
//!
//! ```no_run
//! use batchalign::cache::{CacheBackend, SqliteBackend};
//!
//! # async fn example() -> Result<(), batchalign::cache::CacheError> {
//! let tmp = tempfile::TempDir::new().unwrap();
//! let backend = SqliteBackend::open(Some(tmp.path().to_path_buf())).await?;
//!
//! let data = serde_json::json!({"indexed_timings": []});
//! backend.put("k1", "forced_alignment", "wave2vec-fa-v1", "3.0.0", &data).await?;
//!
//! // Retrieve it (only if the namespace matches).
//! assert_eq!(backend.get("k1", "forced_alignment", "wave2vec-fa-v1").await?, Some(data));
//! // A different namespace is a miss.
//! assert_eq!(backend.get("k1", "forced_alignment", "whisper-fa-v1").await?, None);
//!
//! let stats = backend.stats().await?;
//! assert_eq!(stats.total_entries, 1);
//! # Ok(())
//! # }
//! ```

mod backend;
mod inference_lease;
mod noop;
mod sqlite;
mod tiered;

pub use backend::{CacheBackend, CacheStats};
pub(crate) use inference_lease::InferenceLease;
pub use sqlite::SqliteBackend;
pub use tiered::TieredCacheBackend;

use std::marker::PhantomData;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::chat_ops::CacheTaskName;
use crate::options::UtrEngine;

// ---------------------------------------------------------------------------
// Typed tasks and namespaces
// ---------------------------------------------------------------------------

mod sealed {
    /// Implemented only in this module, for the namespace types listed below,
    /// so no other type can become a cache namespace.
    pub trait Sealed {}
}

/// The identity every row of one cache task is written and read under.
///
/// Sealed: the implementors are exactly the typed identities the task
/// constants in [`tasks`] name. Each type implements [`Self::namespace`] in
/// its own module, from its private field.
pub(crate) trait CacheNamespace: sealed::Sealed {
    /// The namespace bytes the cache stores.
    fn namespace(&self) -> &str;
}

impl sealed::Sealed for crate::engine_reports::FaCacheNamespace {}
impl sealed::Sealed for UtrAsrCacheNamespace {}
impl sealed::Sealed for crate::revai::RevAsrModelRevision {}
impl sealed::Sealed for crate::transcribe::SpeakerEvidenceModelRevision {}
impl sealed::Sealed for crate::transcribe::SpeakerNormalizationRevision {}

impl CacheNamespace for crate::engine_reports::FaCacheNamespace {
    /// The FA engine name the selected worker reported, byte for byte.
    fn namespace(&self) -> &str {
        self.name().as_str()
    }
}

/// One cache task paired with the namespace type its rows are written under.
///
/// The only values are the constants in [`tasks`]; the constructor is private
/// to this module.
pub(crate) struct CacheTask<N> {
    name: CacheTaskName,
    namespace: PhantomData<fn(&N)>,
}

// Written out rather than derived: a derive would bound `N` by the same
// traits, and the namespace types are not all `Copy`.
impl<N> Clone for CacheTask<N> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<N> Copy for CacheTask<N> {}

impl<N> CacheTask<N> {
    const fn new(name: CacheTaskName) -> Self {
        Self {
            name,
            namespace: PhantomData,
        }
    }

    /// The task's wire name, for diagnostics.
    pub(crate) const fn name(self) -> CacheTaskName {
        self.name
    }
}

/// Every cache task, each with the namespace type its rows use.
pub(crate) mod tasks {
    use super::{CacheTask, UtrAsrCacheNamespace};
    use crate::chat_ops::CacheTaskName;

    /// Derived FA timings, namespaced by the reported FA engine.
    pub(crate) const FORCED_ALIGNMENT: CacheTask<crate::engine_reports::FaCacheNamespace> =
        CacheTask::new(CacheTaskName::ForcedAlignment);
    /// Immutable FA worker responses, namespaced by the reported FA engine.
    pub(crate) const FORCED_ALIGNMENT_RAW_EVIDENCE: CacheTask<
        crate::engine_reports::FaCacheNamespace,
    > = CacheTask::new(CacheTaskName::ForcedAlignmentRawEvidence);
    /// Normalized UTR ASR responses, namespaced by the UTR engine.
    pub(crate) const UTR_ASR: CacheTask<UtrAsrCacheNamespace> =
        CacheTask::new(CacheTaskName::UtrAsr);
    /// Raw Rev.AI transcripts, namespaced by the provider revision.
    pub(crate) const REV_ASR_EVIDENCE: CacheTask<crate::revai::RevAsrModelRevision> =
        CacheTask::new(CacheTaskName::RevAsrEvidence);
    /// Raw speaker evidence, namespaced by the speaker model revision.
    pub(crate) const SPEAKER_DIARIZATION_RAW_EVIDENCE: CacheTask<
        crate::transcribe::SpeakerEvidenceModelRevision,
    > = CacheTask::new(CacheTaskName::SpeakerDiarizationRawEvidence);
    /// Derived speaker segments, namespaced by the normalization revision.
    pub(crate) const SPEAKER_DIARIZATION_SEGMENTS: CacheTask<
        crate::transcribe::SpeakerNormalizationRevision,
    > = CacheTask::new(CacheTaskName::SpeakerDiarizationSegments);
}

/// The namespace UTR ASR responses are cached under: the UTR engine's own.
///
/// It used to be the FORCED-ALIGNMENT engine version, so changing the FA model
/// invalidated every UTR ASR response while changing the UTR engine did not.
/// The bytes are `utr-asr-v1:<engine wire name>`, with a version prefix so a
/// later change to what a UTR engine identity includes can move to a new
/// namespace instead of colliding with this one. Written as a match on the
/// closed engine set, so each namespace is a `'static` string and a new engine
/// must state its own.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct UtrAsrCacheNamespace(BuildOwnedNamespace);

impl UtrAsrCacheNamespace {
    /// The namespace for one UTR engine running one pinned plan.
    ///
    /// The engine name alone said only WHICH engine produced a row, never which
    /// weights it produced it with, so a row written before a checkpoint moved
    /// was indistinguishable from one written after. Naming the pinned models
    /// makes that distinction structural: change any model and rows written
    /// under the old one are unreadable rather than silently reused.
    ///
    /// The `utr-asr-v1:` prefix is kept deliberately. W1 already moved this
    /// namespace once in this build, and reusing its prefix folds the identity
    /// into that same move, so the release costs ONE recompute rather than two.
    pub(crate) fn for_pinned_plan(
        engine: &UtrEngine,
        models: &crate::types::worker_v2::AsrRequestedModelsV2,
    ) -> UtrAsrCacheEligibility {
        match models.pinned_namespace_text() {
            Some(pin) => UtrAsrCacheEligibility::Pinned(Self(BuildOwnedNamespace::derived(
                format!("utr-asr-v1:{}:{pin}", engine.as_wire_name()),
            ))),
            None => UtrAsrCacheEligibility::Floating,
        }
    }
}

/// Whether one UTR ASR plan may use the cache at all.
///
/// A closed pair rather than an `Option<UtrAsrCacheNamespace>`, so every caller
/// has to say what it does when a plan cannot be cached instead of reaching for
/// `unwrap_or` and quietly reusing another plan's rows.
pub(crate) enum UtrAsrCacheEligibility {
    /// Every model is pinned, so rows may be written and read under this
    /// namespace.
    Pinned(UtrAsrCacheNamespace),
    /// At least one model floats, so no stored row can promise it came from the
    /// same weights. Such a run infers without touching the cache.
    Floating,
}

impl CacheNamespace for UtrAsrCacheNamespace {
    fn namespace(&self) -> &str {
        self.0.as_str()
    }
}

/// The bytes of a cache namespace this build names for itself, as opposed to
/// [`crate::engine_reports::FaCacheNamespace`], which a worker reports.
///
/// The one representation behind the UTR ASR namespace and the Rev.AI and
/// speaker revisions: a literal, or text derived from literals and from files
/// compiled into the binary. Each of those keeps its own newtype, so a cache
/// task constant still refuses one where another belongs; this only stops the
/// four from each choosing a representation, and from borrowing a
/// worker-version type for bytes no worker ever reported.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BuildOwnedNamespace(std::borrow::Cow<'static, str>);

impl BuildOwnedNamespace {
    /// A namespace written as a literal.
    pub(crate) const fn literal(bytes: &'static str) -> Self {
        Self(std::borrow::Cow::Borrowed(bytes))
    }

    /// A namespace derived when it is needed, from literals and compiled-in
    /// files.
    pub(crate) fn derived(bytes: String) -> Self {
        Self(std::borrow::Cow::Owned(bytes))
    }

    /// The namespace bytes.
    pub(crate) fn as_str(&self) -> &str {
        &self.0
    }
}

static NEXT_CACHE_INSTANCE_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct CacheInstanceId(u64);

impl CacheInstanceId {
    fn fresh() -> Self {
        Self(NEXT_CACHE_INSTANCE_ID.fetch_add(1, Ordering::Relaxed))
    }
}

/// High-level cache wrapper.
///
/// Wraps a `Box<dyn CacheBackend>` and provides factory methods for
/// the supported backends.
pub struct UtteranceCache {
    backend: Box<dyn CacheBackend>,
    instance_id: CacheInstanceId,
}

impl UtteranceCache {
    /// Create a local SQLite-backed cache.
    ///
    /// If `cache_dir` is `None`, uses the platform default:
    /// `~/Library/Caches/batchalign3` on macOS (matching Python's
    /// `platformdirs.user_cache_dir("batchalign3", "batchalign3")`).
    pub async fn sqlite(cache_dir: Option<PathBuf>) -> Result<Self, CacheError> {
        let backend = SqliteBackend::open(cache_dir).await?;
        Ok(Self {
            backend: Box::new(backend),
            instance_id: CacheInstanceId::fresh(),
        })
    }

    /// Create a tiered cache: moka in-memory hot layer + SQLite cold backend.
    ///
    /// - `cache_dir`: SQLite directory (None = platform default).
    /// - `max_hot_entries`: hot-cache capacity (None = 10,000 entries).
    pub async fn tiered(
        cache_dir: Option<PathBuf>,
        max_hot_entries: Option<u64>,
    ) -> Result<Self, CacheError> {
        let cold = SqliteBackend::open(cache_dir).await?;
        let tiered = TieredCacheBackend::new(Box::new(cold), max_hot_entries);
        Ok(Self {
            backend: Box::new(tiered),
            instance_id: CacheInstanceId::fresh(),
        })
    }

    /// Create a no-op cache that always misses.
    ///
    /// Use this for tasks where caching adds overhead without meaningful
    /// benefit (e.g., text NLP tasks where re-inference with warm workers
    /// is faster than SQLite lookups on a large cache). Puts are silently
    /// discarded; gets always return `None`.
    pub fn noop() -> Self {
        Self {
            backend: Box::new(noop::NoopBackend),
            instance_id: CacheInstanceId::fresh(),
        }
    }

    /// Create a cache from an existing backend (for testing or custom backends).
    pub fn from_backend(backend: Box<dyn CacheBackend>) -> Self {
        Self {
            backend,
            instance_id: CacheInstanceId::fresh(),
        }
    }

    pub(super) fn instance_id(&self) -> CacheInstanceId {
        self.instance_id
    }

    /// Read one row of `task`, written under `namespace`.
    pub(crate) async fn get<N: CacheNamespace>(
        &self,
        key: &str,
        task: CacheTask<N>,
        namespace: &N,
    ) -> Result<Option<serde_json::Value>, CacheError> {
        self.backend
            .get(key, task.name.as_str(), namespace.namespace())
            .await
    }

    /// Read many rows of `task`, written under `namespace`.
    pub(crate) async fn get_batch<N: CacheNamespace>(
        &self,
        keys: &[String],
        task: CacheTask<N>,
        namespace: &N,
    ) -> Result<std::collections::HashMap<String, serde_json::Value>, CacheError> {
        self.backend
            .get_batch(keys, task.name.as_str(), namespace.namespace())
            .await
    }

    /// Write one row of `task` under `namespace`, stamped with this build's
    /// crate version.
    pub(crate) async fn put<N: CacheNamespace>(
        &self,
        key: &str,
        task: CacheTask<N>,
        namespace: &N,
        data: &serde_json::Value,
    ) -> Result<(), CacheError> {
        self.backend
            .put(
                key,
                task.name.as_str(),
                namespace.namespace(),
                env!("CARGO_PKG_VERSION"),
                data,
            )
            .await
    }

    /// Write many rows of `task` under `namespace`, stamped with this build's
    /// crate version.
    pub(crate) async fn put_batch<N: CacheNamespace>(
        &self,
        entries: &[(String, serde_json::Value)],
        task: CacheTask<N>,
        namespace: &N,
    ) -> Result<(), CacheError> {
        self.backend
            .put_batch(
                entries,
                task.name.as_str(),
                namespace.namespace(),
                env!("CARGO_PKG_VERSION"),
            )
            .await
    }

    /// Delete rows of `task` whatever namespace they were written under.
    pub async fn delete_batch(
        &self,
        keys: &[String],
        task: CacheTaskName,
    ) -> Result<usize, CacheError> {
        self.backend.delete_batch(keys, task.as_str()).await
    }

    /// Row counts and sizes.
    pub async fn stats(&self) -> Result<CacheStats, CacheError> {
        self.backend.stats().await
    }
}

#[cfg(test)]
mod typed_cache_tests {
    use super::{CacheNamespace, UtrAsrCacheNamespace};
    use crate::options::UtrEngine;
    use crate::types::engines::EngineBackend as _;

    /// The namespace one engine resolves, which must be the pinned arm.
    fn namespace_for(engine: &UtrEngine) -> UtrAsrCacheNamespace {
        let lang = crate::api::LanguageCode3::eng();
        let models = crate::model_manifest::utr_pinned_models(engine, &lang)
            .expect("a UTR engine resolves a pinned composition");
        match UtrAsrCacheNamespace::for_pinned_plan(engine, &models) {
            super::UtrAsrCacheEligibility::Pinned(namespace) => namespace,
            super::UtrAsrCacheEligibility::Floating => {
                panic!("every UTR engine composition is fully pinned in this build")
            }
        }
    }

    /// Each UTR engine's namespace keeps the `utr-asr-v1:<wire name>:` prefix
    /// and then names the models it ran. The prefix is pinned byte for byte, so
    /// rows one run writes are read back by later runs; the model half means a
    /// checkpoint move lands in a different namespace instead of silently
    /// reusing rows produced by other weights.
    #[test]
    fn utr_asr_namespaces_name_the_engine_and_its_pinned_models() {
        for engine in [UtrEngine::RevAi, UtrEngine::Whisper, UtrEngine::HkTencent] {
            let bytes = namespace_for(&engine).namespace().to_owned();
            let prefix = format!("utr-asr-v1:{}:", engine.wire_name());
            assert!(
                bytes.starts_with(&prefix),
                "{bytes} should start with {prefix}"
            );
            assert!(
                bytes.contains('@'),
                "the namespace must name a revision, got {bytes}"
            );
        }
        // Two engines never share a namespace, which is what stops one
        // engine's rows being read back for another.
        assert_ne!(
            namespace_for(&UtrEngine::RevAi).namespace(),
            namespace_for(&UtrEngine::Whisper).namespace()
        );
    }

    /// The typed read finds what the typed write stored, and a different
    /// namespace of the same type is a miss.
    #[tokio::test]
    async fn a_row_is_read_back_only_under_its_namespace() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let cache = super::UtteranceCache::sqlite(Some(tempdir.path().join("cache")))
            .await
            .expect("cache");
        let data = serde_json::json!({"segments": []});
        let whisper = namespace_for(&UtrEngine::Whisper);
        cache
            .put("k", super::tasks::UTR_ASR, &whisper, &data)
            .await
            .expect("write");
        assert_eq!(
            cache
                .get("k", super::tasks::UTR_ASR, &whisper)
                .await
                .expect("read"),
            Some(data)
        );
        let rev = namespace_for(&UtrEngine::RevAi);
        assert_eq!(
            cache
                .get("k", super::tasks::UTR_ASR, &rev)
                .await
                .expect("read"),
            None
        );
    }
}

/// Cache errors.
#[derive(Debug, thiserror::Error)]
pub enum CacheError {
    /// Database operation failed.
    #[error("database error: {0}")]
    Database(#[from] sqlx::Error),

    /// Database migration failed.
    #[error("migration error: {0}")]
    Migration(#[from] sqlx::migrate::MigrateError),

    /// JSON serialization or deserialization failed.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Filesystem I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Platform cache directory could not be determined.
    #[error("Cache directory not found")]
    NoCacheDir,
}
