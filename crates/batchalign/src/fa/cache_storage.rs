//! Physical storage for FA evidence layers; semantic request identity stays unchanged.
//!
//! SQLite's primary key is the key alone, not (key, task). Keep raw evidence at
//! its established semantic address and derive a disjoint timing-layer address.

use super::{AdmittedCachedFaTimings, FaServices};
use crate::cache::{
    UtteranceCache,
    tasks::{FORCED_ALIGNMENT, FORCED_ALIGNMENT_RAW_EVIDENCE},
};
use crate::chat_ops::fa::WordTiming;
use std::collections::HashMap;
use tracing::warn;

/// Only this storage owner can make or use a physical derived-timing address.
/// The semantic key still identifies the admitted request inside its evidence.
struct DerivedFaStorageKey<'key> {
    semantic: &'key str,
    physical: String,
}

impl<'key> DerivedFaStorageKey<'key> {
    fn of(semantic: &'key str) -> Self {
        Self {
            semantic,
            physical: format!("fa-derived-v1:{semantic}"),
        }
    }
}

pub(super) async fn read_values(
    keys: &[String],
    services: FaServices<'_>,
) -> (
    HashMap<String, serde_json::Value>,
    HashMap<String, serde_json::Value>,
) {
    if keys.is_empty() {
        return (HashMap::new(), HashMap::new());
    }
    let addresses: Vec<_> = keys
        .iter()
        .map(|key| DerivedFaStorageKey::of(key))
        .collect();
    let derived_keys: Vec<_> = addresses
        .iter()
        .map(|address| address.physical.clone())
        .collect();
    let (derived, raw) = tokio::join!(
        services.pipeline.cache.get_batch(
            &derived_keys,
            FORCED_ALIGNMENT,
            services.cache_namespace
        ),
        services.pipeline.cache.get_batch(
            keys,
            FORCED_ALIGNMENT_RAW_EVIDENCE,
            services.cache_namespace
        ),
    );
    let mut derived = match derived {
        Ok(values) => values,
        Err(error) => {
            warn!(%error, "FA derived cache unavailable; retain explicit inference misses");
            HashMap::new()
        }
    };
    // An older timing-only row can still live at the semantic address. Read it
    // only for missing new addresses, without rewriting or deleting evidence.
    let legacy_keys: Vec<_> = addresses
        .iter()
        .filter(|address| !derived.contains_key(&address.physical))
        .map(|address| address.semantic.to_owned())
        .collect();
    let mut legacy = if legacy_keys.is_empty() {
        HashMap::new()
    } else {
        match services
            .pipeline
            .cache
            .get_batch(&legacy_keys, FORCED_ALIGNMENT, services.cache_namespace)
            .await
        {
            Ok(values) => values,
            Err(error) => {
                warn!(%error, "Legacy FA derived cache unavailable; retain explicit inference misses");
                HashMap::new()
            }
        }
    };
    let derived = addresses
        .into_iter()
        .filter_map(|address| {
            derived
                .remove(&address.physical)
                .or_else(|| legacy.remove(address.semantic))
                .map(|value| (address.semantic.to_owned(), value))
        })
        .collect();
    let raw = match raw {
        Ok(values) => values,
        Err(error) => {
            warn!(%error, "FA raw cache unavailable; retain explicit inference misses");
            HashMap::new()
        }
    };
    (derived, raw)
}

/// Consuming replayable evidence supplies its admitted semantic key and engine
/// namespace. A caller cannot independently choose either cache identity.
/// Cache writes remain best-effort; failures do not invalidate this run's timings.
pub(super) async fn store_unit_evidence(
    cache: &UtteranceCache,
    timings: &[Option<WordTiming>],
    raw_evidence: super::raw_evidence::ReplayableFaRawEvidence,
) {
    let key = raw_evidence.cache_key().as_str().to_owned();
    let namespace = raw_evidence.request_engine_version().clone();
    let derived_key = DerivedFaStorageKey::of(&key);
    match AdmittedCachedFaTimings::encode_from_raw(timings.to_vec(), &raw_evidence) {
        Ok(cache_data) => {
            if let Err(error) = cache
                .put_batch(
                    &[(derived_key.physical, cache_data)],
                    FORCED_ALIGNMENT,
                    &namespace,
                )
                .await
            {
                warn!(error = %error, "Failed to cache derived FA evidence (non-fatal)");
            }
        }
        Err(error) => {
            warn!(error = %error, "Failed to encode derived FA evidence (non-fatal)");
        }
    }
    match serde_json::to_value(raw_evidence) {
        Ok(cache_data) => {
            if let Err(error) = cache
                .put_batch(
                    &[(key, cache_data)],
                    FORCED_ALIGNMENT_RAW_EVIDENCE,
                    &namespace,
                )
                .await
            {
                warn!(error = %error, "Failed to cache raw FA evidence (non-fatal)");
            }
        }
        Err(error) => {
            warn!(error = %error, "Failed to serialize raw FA evidence (non-fatal)");
        }
    }
}
