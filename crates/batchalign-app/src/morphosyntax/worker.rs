//! Worker dispatch for morphosyntax inference.

use std::sync::Arc;

use crate::api::LanguageCode3;
use crate::error::ServerError;
use crate::infer_retry::dispatch_execute_v2_with_retry_and_progress;
use crate::worker::artifacts_v2::PreparedArtifactRuntimeV2;
use crate::worker::pool::WorkerPool;
use crate::worker::text_request_v2::{PreparedTextRequestIdsV2, build_morphosyntax_request_v2};
use crate::worker::text_result_v2::parse_morphosyntax_result_v2;
use batchalign_chat_ops::morphosyntax::{BatchItemWithPosition, MwtDict, stanza_raw};
use batchalign_chat_ops::nlp::UdResponse;
use tracing::{info, warn};

/// Send batch items to workers for NLP inference via batched `execute_v2`.
///
/// When the batch is large enough and the pool allows multiple workers per
/// language key, the items are split into chunks and dispatched concurrently
/// to separate workers.  This is transparent to callers — the returned
/// `Vec<UdResponse>` is always parallel to the input `items` slice.
pub(crate) async fn infer_batch(
    pool: &WorkerPool,
    items: &[BatchItemWithPosition],
    lang: &LanguageCode3,
    mwt: &MwtDict,
    retokenize: bool,
    progress_tx: Option<&tokio::sync::mpsc::Sender<crate::types::worker_v2::ProgressEventV2>>,
) -> Result<Vec<UdResponse>, ServerError> {
    // Python workers emit `stage="stanza_processing"` on every progress
    // event (see `batchalign/worker/_protocol.py::write_progress_event`).
    // The drain loop in `runner/dispatch/infer_batched.rs` keys
    // `BatchInferProgress` by `event.stage`, so without a language-aware
    // rewrite every language collapses into one bucket and stall
    // detection goes blind. The tagger below rewrites `event.stage` to
    // the language code for the duration of this batch. Only spawned
    // when the caller actually wants progress updates.
    let tagger = ProgressTagger::install(progress_tx, lang);
    let inner_tx = tagger.sender();

    let num_chunks = compute_chunk_count(items.len(), pool.max_workers_per_key());

    let result = if num_chunks <= 1 {
        infer_batch_single(pool, items, lang, mwt, retokenize, inner_tx).await
    } else {
        let chunk_size = items.len().div_ceil(num_chunks);
        let chunks: Vec<&[BatchItemWithPosition]> = items.chunks(chunk_size).collect();
        info!(
            items = items.len(),
            chunks = chunks.len(),
            chunk_size,
            lang = %lang,
            "Splitting morphosyntax batch across workers"
        );
        let futures: Vec<_> = chunks
            .iter()
            .map(|chunk| infer_batch_single(pool, chunk, lang, mwt, retokenize, inner_tx))
            .collect();
        let outcomes = futures::future::join_all(futures).await;
        let mut all = Vec::with_capacity(items.len());
        for outcome in outcomes {
            all.extend(outcome?);
        }
        Ok(all)
    };

    tagger.close().await;
    result
}

/// Owns the inner mpsc channel + forwarder task that rewrites
/// `event.stage` on every progress event. Explicit struct so the
/// ownership boundary is visible (inner channel lives exactly for the
/// duration of a batch; outer channel is borrowed from the caller).
struct ProgressTagger {
    inner_tx: Option<tokio::sync::mpsc::Sender<crate::types::worker_v2::ProgressEventV2>>,
    handle: Option<tokio::task::JoinHandle<()>>,
}

impl ProgressTagger {
    fn install(
        outer: Option<&tokio::sync::mpsc::Sender<crate::types::worker_v2::ProgressEventV2>>,
        lang: &LanguageCode3,
    ) -> Self {
        let outer = match outer {
            Some(tx) => tx.clone(),
            None => {
                return Self {
                    inner_tx: None,
                    handle: None,
                };
            }
        };
        let (inner_tx, mut inner_rx) =
            tokio::sync::mpsc::channel::<crate::types::worker_v2::ProgressEventV2>(64);
        let tag: Arc<str> = Arc::from(lang.as_ref());
        let handle = tokio::spawn(async move {
            while let Some(mut event) = inner_rx.recv().await {
                event.stage = tag.as_ref().to_string();
                if outer.send(event).await.is_err() {
                    break;
                }
            }
        });
        Self {
            inner_tx: Some(inner_tx),
            handle: Some(handle),
        }
    }

    fn sender(
        &self,
    ) -> Option<&tokio::sync::mpsc::Sender<crate::types::worker_v2::ProgressEventV2>> {
        self.inner_tx.as_ref()
    }

    async fn close(self) {
        drop(self.inner_tx);
        if let Some(handle) = self.handle {
            let _ = handle.await;
        }
    }
}

/// Dispatch a single chunk of batch items to one worker.
///
/// This is the original `infer_batch` body, extracted so it can be called
/// once (fast path) or N times concurrently (chunked path).
async fn infer_batch_single(
    pool: &WorkerPool,
    items: &[BatchItemWithPosition],
    lang: &LanguageCode3,
    mwt: &MwtDict,
    retokenize: bool,
    progress_tx: Option<&tokio::sync::mpsc::Sender<crate::types::worker_v2::ProgressEventV2>>,
) -> Result<Vec<UdResponse>, ServerError> {
    let payload_items: Vec<_> = items.iter().map(|(_, _, item, _)| item.clone()).collect();

    let artifacts = PreparedArtifactRuntimeV2::new("morphosyntax_v2").map_err(|error| {
        ServerError::Validation(format!(
            "failed to create morphosyntax V2 artifact runtime: {error}"
        ))
    })?;
    let request_ids = PreparedTextRequestIdsV2::for_task("morphosyntax");
    let request = build_morphosyntax_request_v2(
        artifacts.store(),
        &request_ids,
        lang,
        &payload_items,
        mwt,
        retokenize,
    )
    .map_err(|error| {
        ServerError::Validation(format!(
            "failed to build morphosyntax V2 worker request: {error}"
        ))
    })?;

    info!(
        num_items = items.len(),
        lang = %lang,
        "Dispatching morphosyntax execute_v2 batch"
    );

    let response =
        dispatch_execute_v2_with_retry_and_progress(pool, lang, &request, progress_tx).await?;
    let result = parse_morphosyntax_result_v2(&response).map_err(|error| {
        ServerError::Validation(format!("invalid morphosyntax V2 result: {error}"))
    })?;
    if result.items.len() != items.len() {
        return Err(ServerError::Validation(format!(
            "morphosyntax V2 returned {} items for {} requests",
            result.items.len(),
            items.len()
        )));
    }

    let mut ud_responses = Vec::with_capacity(result.items.len());
    for (i, item_result) in result.items.iter().enumerate() {
        if let Some(error) = &item_result.error {
            warn!(item = i, error = %error, "Infer error for item (using empty response)");
            ud_responses.push(UdResponse {
                sentences: Vec::new(),
            });
            continue;
        }

        if let Some(raw_sentences) = &item_result.raw_sentences {
            let ud = stanza_raw::parse_raw_stanza_output(raw_sentences).map_err(|error| {
                // Include the words sent, structured diagnostics, and raw
                // Stanza output so the failure is diagnosable without replay.
                let words_sent = &payload_items[i].words;
                let diagnostics = stanza_raw::diagnose_parse_failure(raw_sentences);
                let diag_str = if diagnostics.is_empty() {
                    "no structural issues detected by diagnostics".to_string()
                } else {
                    diagnostics
                        .iter()
                        .map(|d| d.to_string())
                        .collect::<Vec<_>>()
                        .join("; ")
                };
                let raw_json = serde_json::to_string(raw_sentences)
                    .unwrap_or_else(|_| "<serialization failed>".into());
                warn!(
                    item = i,
                    words = ?words_sent,
                    diagnostics = %diag_str,
                    raw_stanza_output = %raw_json,
                    %error,
                    "Stanza output parse failure — full diagnostics logged"
                );
                ServerError::Validation(format!(
                    "Failed to parse raw Stanza output for item {i} \
                     (words: {words_sent:?}): {error}. Diagnostics: {diag_str}"
                ))
            })?;
            ud_responses.push(ud);
            continue;
        }

        warn!(
            item = i,
            "Morphosyntax V2 returned no raw_sentences and no error (using empty response)"
        );
        ud_responses.push(UdResponse {
            sentences: Vec::new(),
        });
    }

    Ok(ud_responses)
}

/// Minimum items per chunk.  Below this threshold, Stanza's per-batch
/// overhead (model forward-pass setup, tokenizer warmup) dominates and
/// splitting provides no throughput benefit.
const MIN_CHUNK_SIZE: usize = 30;

/// Compute how many worker chunks to split a language batch into.
///
/// Returns 1 (no split) when:
/// - Fewer than `MIN_CHUNK_SIZE` items (splitting not worthwhile).
/// - `max_workers` is 1 (only one worker slot available).
///
/// Otherwise returns `min(item_count / MIN_CHUNK_SIZE, max_workers)`.
fn compute_chunk_count(item_count: usize, max_workers: usize) -> usize {
    if item_count < MIN_CHUNK_SIZE || max_workers <= 1 {
        return 1;
    }
    (item_count / MIN_CHUNK_SIZE).clamp(1, max_workers)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_chunk_count_below_minimum_returns_one() {
        assert_eq!(compute_chunk_count(0, 4), 1);
        assert_eq!(compute_chunk_count(1, 4), 1);
        assert_eq!(compute_chunk_count(29, 4), 1);
    }

    #[test]
    fn compute_chunk_count_at_minimum_returns_one() {
        // 30 / 30 = 1
        assert_eq!(compute_chunk_count(30, 4), 1);
    }

    #[test]
    fn compute_chunk_count_scales_with_items() {
        assert_eq!(compute_chunk_count(60, 4), 2);
        assert_eq!(compute_chunk_count(90, 4), 3);
        assert_eq!(compute_chunk_count(120, 4), 4);
    }

    #[test]
    fn compute_chunk_count_clamped_by_max_workers() {
        // 2000 / 30 = 66, but max_workers = 4
        assert_eq!(compute_chunk_count(2000, 4), 4);
        assert_eq!(compute_chunk_count(500, 2), 2);
        assert_eq!(compute_chunk_count(500, 8), 8);
    }

    #[test]
    fn compute_chunk_count_single_worker_always_one() {
        assert_eq!(compute_chunk_count(2000, 1), 1);
        assert_eq!(compute_chunk_count(60, 1), 1);
    }

    #[test]
    fn compute_chunk_count_zero_workers_returns_one() {
        // Defensive: max_workers=0 should not panic
        assert_eq!(compute_chunk_count(100, 0), 1);
    }
}
