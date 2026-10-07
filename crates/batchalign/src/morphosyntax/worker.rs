//! Worker dispatch for morphosyntax inference.

use std::collections::HashMap;

use super::AnalysisUnavailable;
use super::identity::AdmittedMorphosyntaxResponse;
use crate::api::LanguageCode3;
use crate::chat_ops::morphosyntax_ops::{MorphosyntaxBatchItem, MwtDict};
use crate::error::ServerError;
use crate::execution::morphotag::progress::BackendProgressPort;
use crate::infer_retry::{Cancellation, dispatch_execute_v2_with_retry_and_progress};
use crate::runner::util::batch_progress::BatchChunkIndex;
use crate::types::worker_v2::MorphosyntaxItemResultV2;
use crate::worker::artifacts_v2::PreparedArtifactRuntimeV2;
use crate::worker::pool::WorkerPool;
use crate::worker::text_request_v2::{PreparedTextRequestIdsV2, build_morphosyntax_request_v2};
use crate::worker::text_result_v2::parse_morphosyntax_result_v2;
use batchalign_transform::morphosyntax::{diagnose_parse_failure, parse_raw_stanza_output};
use tracing::{info, warn};

#[derive(Debug, Clone, PartialEq, Eq)]
struct LanguageBatchGroup {
    lang: LanguageCode3,
    indices: Vec<usize>,
}

/// Owns the exact grouped payload view whose effective languages the runtime
/// admitted. All groups are checked before any worker can be dispatched.
struct DispatchPlan<'a, T> {
    pool: &'a WorkerPool,
    items: &'a [T],
    groups: Vec<LanguageBatchGroup>,
}

impl<'a, T: AsRef<MorphosyntaxBatchItem>> DispatchPlan<'a, T> {
    fn admit(
        pool: &'a WorkerPool,
        items: &'a [T],
        fallback_lang: &LanguageCode3,
    ) -> Result<Self, ServerError> {
        let groups = language_groups_for_items(items, fallback_lang)?;
        for group in &groups {
            AnalysisUnavailable::admit_effective(&group.lang, pool.stanza_registry())?;
        }
        Ok(Self {
            pool,
            items,
            groups,
        })
    }
}

fn language_groups_for_items<T: AsRef<MorphosyntaxBatchItem>>(
    items: &[T],
    fallback_lang: &LanguageCode3,
) -> Result<Vec<LanguageBatchGroup>, ServerError> {
    let mut groups: Vec<LanguageBatchGroup> = Vec::new();
    let mut positions: HashMap<String, usize> = HashMap::new();

    for (idx, item) in items.iter().enumerate() {
        let item = item.as_ref();
        let effective_lang = if item.lang.as_ref().is_empty() {
            fallback_lang.clone()
        } else {
            LanguageCode3::try_new(item.lang.as_ref()).map_err(|error| {
                ServerError::Validation(format!(
                    "morphotag batch item has invalid language '{}': {error}",
                    item.lang.as_ref()
                ))
            })?
        };

        let key = effective_lang.as_ref().to_string();
        if let Some(group_idx) = positions.get(&key).copied() {
            groups[group_idx].indices.push(idx);
        } else {
            positions.insert(key, groups.len());
            groups.push(LanguageBatchGroup {
                lang: effective_lang,
                indices: vec![idx],
            });
        }
    }

    Ok(groups)
}

/// Send batch items to workers for NLP inference via batched `execute_v2`.
///
/// When the batch is large enough and the pool allows multiple workers per
/// language key, the items are split into chunks and dispatched concurrently
/// to separate workers.  This is transparent to callers, the returned
/// responses are always parallel to the input `items` slice, each admitted
/// with the model that produced it.
pub(crate) async fn infer_batch<T: AsRef<MorphosyntaxBatchItem> + Sync>(
    pool: &WorkerPool,
    items: &[T],
    lang: &LanguageCode3,
    mwt: &MwtDict,
    retokenize: bool,
    progress: Option<&BackendProgressPort>,
    cancellation: Cancellation<'_>,
) -> Result<Vec<AdmittedMorphosyntaxResponse>, ServerError> {
    let item_results = DispatchPlan::admit(pool, items, lang)?
        .infer(mwt, retokenize, progress, cancellation)
        .await?;
    admit_worker_responses(item_results)
}

/// Retain the full per-item failure inventory while refusing completion.
/// These errors describe model output, never the submitted CHAT's validity.
fn admit_worker_responses(
    item_results: Vec<Result<AdmittedMorphosyntaxResponse, String>>,
) -> Result<Vec<AdmittedMorphosyntaxResponse>, ServerError> {
    let item_results: Vec<_> = item_results
        .into_iter()
        // Per-item messages retain the worker's failure or native rejection
        // of its output. Morphotag has no command-specific empty-result
        // outcome of the kind translate carries.
        .map(|item| item.map_err(crate::text_batch::EngineItemFailure::EngineReported))
        .collect();
    crate::text_batch::unwrap_per_item_results("morphotag", item_results).map_err(|err| {
        ServerError::OutputAdmission {
            command: crate::api::ReleasedCommand::Morphotag,
            details: crate::error::OutputAdmissionRefusal::unestablished(err.to_string()),
        }
    })
}

impl<T: AsRef<MorphosyntaxBatchItem> + Sync> DispatchPlan<'_, T> {
    /// Dispatch is reachable only on an admitted plan, not a raw payload slice.
    async fn infer(
        self,
        mwt: &MwtDict,
        retokenize: bool,
        progress: Option<&BackendProgressPort>,
        cancellation: Cancellation<'_>,
    ) -> Result<Vec<Result<AdmittedMorphosyntaxResponse, String>>, ServerError> {
        let Self {
            pool,
            items,
            groups,
        } = self;
        if items.is_empty() {
            return Ok(Vec::new());
        }
        if let [group] = groups.as_slice() {
            return infer_batch_homogeneous(
                pool,
                items,
                &group.lang,
                mwt,
                retokenize,
                progress,
                cancellation,
            )
            .await;
        }

        info!(
            items = items.len(),
            dispatched_groups = groups.len(),
            "Dispatching admitted morphosyntax batch by effective utterance language"
        );

        let mut merged: Vec<Option<Result<AdmittedMorphosyntaxResponse, String>>> =
            vec![None; items.len()];

        for group in groups {
            let group_items: Vec<&T> = group.indices.iter().map(|&idx| &items[idx]).collect();
            let responses = infer_batch_homogeneous(
                pool,
                &group_items,
                &group.lang,
                mwt,
                retokenize,
                progress,
                cancellation,
            )
            .await?;
            for (original_idx, response) in group.indices.into_iter().zip(responses) {
                merged[original_idx] = Some(response);
            }
        }

        merged
            .into_iter()
            .map(|response| {
                response.ok_or_else(|| ServerError::OutputAdmission {
                    command: crate::api::ReleasedCommand::Morphotag,
                    details: crate::error::OutputAdmissionRefusal::unestablished(
                        "mixed-language dispatch returned incomplete results",
                    ),
                })
            })
            .collect()
    }
}

async fn infer_batch_homogeneous<T: AsRef<MorphosyntaxBatchItem> + Sync>(
    pool: &WorkerPool,
    items: &[T],
    lang: &LanguageCode3,
    mwt: &MwtDict,
    retokenize: bool,
    progress: Option<&BackendProgressPort>,
    cancellation: Cancellation<'_>,
) -> Result<Vec<Result<AdmittedMorphosyntaxResponse, String>>, ServerError> {
    // Progress reporting for this batch.
    //
    // The Python backend emits `progress_v2` events carrying `completed` /
    // `total` for the request it is working on, at most one per second (see
    // `batchalign/worker/_text_v2.py`). It hard-codes `stage =
    // "stanza_processing"` on every one of them, so the wire event says nothing
    // about WHICH work it describes. Provenance therefore comes from here,
    // where the language and the chunk are known for certain, and is attached
    // as typed fields rather than by rewriting `stage` in flight (what the
    // 2026-04 code did, which is how a field named "stage" came to carry a
    // language code by an unenforced convention).
    //
    // One bridge PER CHUNK, not per batch. `MIN_CHUNK_SIZE` splits a language
    // group across workers whenever it holds 60 items or more, each chunk being
    // its own request with its own counts; aggregating those by language alone
    // is what displayed `453/274` (165%) before this feature was removed on
    // 2026-05-03. The chunk index is the stable key, because a retry reissues
    // the same chunk under a fresh request id. Full reasoning:
    // `crate::runner::util::batch_progress`.
    // Declare this group's work BEFORE dispatching it, so the file has an exact
    // denominator from the start. Inferring it from chunk reports makes a file
    // whose first chunk finished look complete while its other chunks have not
    // started, which silently suppressed every per-file update.
    if let Some(port) = progress {
        port.declare_group_total(lang, crate::api::UtteranceCount(items.len() as u64));
    }

    let num_chunks = compute_chunk_count(
        items.len(),
        pool.max_workers_per_key_for(crate::worker::WorkerProfile::Stanza),
    );

    if num_chunks <= 1 {
        let bridge = ChunkProgressBridge::install(progress, lang, BatchChunkIndex(0));
        let result = infer_batch_single(
            pool,
            items,
            lang,
            mwt,
            retokenize,
            bridge.sender(),
            cancellation,
        )
        .await;
        bridge.close().await;
        return result;
    }

    let chunk_size = items.len().div_ceil(num_chunks);
    let chunks: Vec<&[T]> = items.chunks(chunk_size).collect();
    info!(
        items = items.len(),
        chunks = chunks.len(),
        chunk_size,
        lang = %lang,
        "Splitting morphosyntax batch across workers"
    );
    let futures: Vec<_> = chunks
        .iter()
        .enumerate()
        .map(|(index, chunk)| async move {
            // Each chunk reports under its own index, so the ledger can sum
            // chunk totals instead of letting the last report win.
            let bridge = ChunkProgressBridge::install(
                progress,
                lang,
                BatchChunkIndex(u32::try_from(index).unwrap_or(u32::MAX)),
            );
            let result = infer_batch_single(
                pool,
                chunk,
                lang,
                mwt,
                retokenize,
                bridge.sender(),
                cancellation,
            )
            .await;
            bridge.close().await;
            result
        })
        .collect();
    let outcomes = futures::future::join_all(futures).await;
    let mut all = Vec::with_capacity(items.len());
    for outcome in outcomes {
        all.extend(outcome?);
    }
    Ok(all)
}

/// Bridges one chunk's wire progress events into typed domain reports.
///
/// Owns an inner mpsc channel plus the forwarder task that drains it. The
/// inner channel lives exactly as long as one chunk's dispatch; the outer
/// [`BackendProgressPort`] is borrowed from the caller and knows which file
/// the work belongs to. An explicit struct so that ownership boundary is
/// visible, and so `close()` can await the forwarder rather than dropping
/// reports still in flight.
///
/// Installs nothing when the caller passes no port, which is the case for
/// every non-job caller (the CLI's direct path, `compare`'s internal
/// morphotag, tests): no channel, no task, no cost.
struct ChunkProgressBridge {
    inner_tx: Option<tokio::sync::mpsc::Sender<crate::types::worker_v2::ProgressEventV2>>,
    handle: Option<tokio::task::JoinHandle<()>>,
}

impl ChunkProgressBridge {
    fn install(
        port: Option<&BackendProgressPort>,
        lang: &LanguageCode3,
        chunk: BatchChunkIndex,
    ) -> Self {
        let Some(port) = port.cloned() else {
            return Self {
                inner_tx: None,
                handle: None,
            };
        };
        let (inner_tx, mut inner_rx) =
            tokio::sync::mpsc::channel::<crate::types::worker_v2::ProgressEventV2>(64);
        let lang = lang.clone();
        let handle = tokio::spawn(async move {
            while let Some(event) = inner_rx.recv().await {
                port.report(&lang, chunk, &event);
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
async fn infer_batch_single<T: AsRef<MorphosyntaxBatchItem> + Sync>(
    pool: &WorkerPool,
    items: &[T],
    lang: &LanguageCode3,
    mwt: &MwtDict,
    retokenize: bool,
    progress_tx: Option<&tokio::sync::mpsc::Sender<crate::types::worker_v2::ProgressEventV2>>,
    cancellation: Cancellation<'_>,
) -> Result<Vec<Result<AdmittedMorphosyntaxResponse, String>>, ServerError> {
    let payload_items: Vec<MorphosyntaxBatchItem> =
        items.iter().map(|item| item.as_ref().clone()).collect();

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

    let response = dispatch_execute_v2_with_retry_and_progress(
        pool,
        lang,
        &request,
        progress_tx,
        cancellation,
    )
    .await?;
    let result =
        parse_morphosyntax_result_v2(response).map_err(|error| ServerError::OutputAdmission {
            command: crate::api::ReleasedCommand::Morphotag,
            details: crate::error::OutputAdmissionRefusal::unestablished(format!(
                "invalid morphosyntax V2 result: {error}"
            )),
        })?;
    if result.items.len() != items.len() {
        return Err(ServerError::OutputAdmission {
            command: crate::api::ReleasedCommand::Morphotag,
            details: crate::error::OutputAdmissionRefusal::unestablished(format!(
                "morphosyntax V2 returned {} items for {} requests",
                result.items.len(),
                items.len(),
            )),
        });
    }

    // Each item is one of three outcomes, and each outcome carries exactly
    // what it needs: an analysis carries its model, a wordless item carries no
    // identity, and a failure carries its error. There is no "analysis
    // without a model" or "neither result nor error" case left to reject.
    let mut ud_responses = Vec::with_capacity(result.items.len());
    for (i, item_result) in result.items.into_iter().enumerate() {
        let (raw_sentences, model, repairs) = match item_result {
            MorphosyntaxItemResultV2::Failed { error } => {
                ud_responses.push(Err(error));
                continue;
            }
            MorphosyntaxItemResultV2::NoWords => {
                ud_responses.push(Ok(AdmittedMorphosyntaxResponse::no_words()));
                continue;
            }
            MorphosyntaxItemResultV2::Analyzed {
                raw_sentences,
                model,
                repairs,
            } => (raw_sentences, model, repairs),
        };
        match parse_raw_stanza_output(&raw_sentences) {
            Ok(ud) => ud_responses.push(
                AdmittedMorphosyntaxResponse::from_worker(ud, model, repairs)
                    .map_err(|error| format!("worker item {i}: {error}")),
            ),
            Err(error) => {
                // Log full diagnostics so the failure is debuggable
                // without a replay, then surface as a per-item Err
                // so the cross-file driver can attribute the
                // failure back to the file that contributed this
                // item: matches the BA2 "one bad utterance
                // abandons the file" semantics.
                let words_sent: Vec<&str> = payload_items[i]
                    .words()
                    .iter()
                    .map(|word| word.text().as_str())
                    .collect();
                let diagnostics = diagnose_parse_failure(&raw_sentences);
                let diag_str = if diagnostics.is_empty() {
                    "no structural issues detected by diagnostics".to_string()
                } else {
                    diagnostics
                        .iter()
                        .map(|d| d.to_string())
                        .collect::<Vec<_>>()
                        .join("; ")
                };
                let raw_json = serde_json::to_string(&raw_sentences)
                    .unwrap_or_else(|_| "<serialization failed>".into());
                warn!(
                    item = i,
                    words = ?words_sent,
                    diagnostics = %diag_str,
                    raw_stanza_output = %raw_json,
                    %error,
                    "Stanza output parse failure: full diagnostics logged"
                );
                ud_responses.push(Err(format!(
                    "Failed to parse raw Stanza output for item {i} \
                     (words: {words_sent:?}): {error}. Diagnostics: {diag_str}"
                )));
            }
        }
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
    use talkbank_model::Span;
    use talkbank_model::Terminator;

    fn batch_item(lang: &str) -> MorphosyntaxBatchItem {
        MorphosyntaxBatchItem::new(
            Vec::new(),
            Terminator::Period { span: Span::DUMMY },
            talkbank_model::model::LanguageCode::new(lang).expect("valid test language code"),
        )
    }

    #[test]
    fn worker_response_failure_inventory_is_system_not_invalid_chat() {
        let error = admit_worker_responses(vec![
            Err("worker item 0: one payload received 2 sentences".into()),
            Ok(AdmittedMorphosyntaxResponse::no_words()),
            Err("worker item 2: malformed dependency result".into()),
        ])
        .expect_err("any failed item refuses completion");
        assert!(matches!(error, ServerError::OutputAdmission { .. }));
        assert_eq!(
            crate::runner::util::classify_server_error(&error),
            crate::scheduling::FailureCategory::System
        );
        let detail = error.to_string();
        assert!(detail.contains("2 sentences"), "{detail}");
        assert!(detail.contains("malformed dependency result"), "{detail}");
        #[cfg(feature = "server")]
        assert_eq!(
            axum::response::IntoResponse::into_response(error).status(),
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
        );
        assert!(admit_worker_responses(vec![Ok(AdmittedMorphosyntaxResponse::no_words())]).is_ok());
    }

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

    #[test]
    fn language_groups_for_items_uses_per_item_language_and_preserves_indices() {
        let items = vec![batch_item("eng"), batch_item("spa"), batch_item("eng")];
        let groups = language_groups_for_items(&items, &LanguageCode3::eng()).unwrap();
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].lang, LanguageCode3::eng());
        assert_eq!(groups[0].indices, vec![0, 2]);
        assert_eq!(groups[1].lang.as_ref(), "spa");
        assert_eq!(groups[1].indices, vec![1]);
    }

    /// Registry double, not a model download or a timing-dependent test.
    fn registry_supporting(langs: &[&str]) -> crate::stanza_registry::StanzaRegistry {
        use crate::types::worker::StanzaLanguageProcessors;
        use std::collections::BTreeMap;
        let mut caps = BTreeMap::new();
        for &iso3 in langs {
            caps.insert(
                iso3.to_string(),
                StanzaLanguageProcessors {
                    alpha2: iso3.chars().take(2).collect(),
                    processors: ["tokenize", "pos", "lemma", "depparse"]
                        .into_iter()
                        .map(String::from)
                        .collect(),
                },
            );
        }
        crate::stanza_registry::StanzaRegistry::from_capabilities(&caps)
    }

    #[test]
    fn dispatch_plan_binds_supported_groups_to_their_exact_payloads() {
        let items = vec![batch_item("eng"), batch_item("cym"), batch_item("eng")];
        let registry = registry_supporting(&["eng", "cym"]);
        let pool = WorkerPool::with_test_stanza_registry(registry);
        let plan = DispatchPlan::admit(&pool, &items, &LanguageCode3::eng()).unwrap();
        assert!(std::ptr::eq(plan.pool, &pool));
        assert!(std::ptr::eq(plan.items, items.as_slice()));
        assert_eq!(
            plan.groups
                .iter()
                .map(|g| g.lang.as_ref())
                .collect::<Vec<_>>(),
            vec!["eng", "cym"]
        );
        assert_eq!(plan.groups[0].indices, vec![0, 2]);
        assert_eq!(plan.groups[1].indices, vec![1]);
    }

    #[test]
    fn unsupported_effective_language_cannot_receive_a_dispatch_plan() {
        let items = vec![batch_item("eng"), batch_item("que")];
        let registry = registry_supporting(&["eng"]);
        let pool = WorkerPool::with_test_stanza_registry(registry);
        let Err(ServerError::AnalysisUnavailable(error)) =
            DispatchPlan::admit(&pool, &items, &LanguageCode3::eng())
        else {
            panic!("a mixed plan must refuse before dispatching even its supported group");
        };
        assert_eq!(error.language().as_ref(), "que");
        assert_eq!(
            error.reason(),
            super::super::AnalysisUnavailableReason::ProcessorsUnavailable
        );
    }

    #[tokio::test]
    async fn missing_registry_refuses_at_transport_before_any_worker() {
        let pool = WorkerPool::new(crate::worker::pool::PoolConfig::default());
        let items = vec![batch_item("eng")];
        let error = infer_batch(
            &pool,
            &items,
            &LanguageCode3::eng(),
            &MwtDict::default(),
            false,
            None,
            Cancellation::NotWired {
                reason: "capability refusal test",
            },
        )
        .await
        .unwrap_err();
        let ServerError::AnalysisUnavailable(error) = error else {
            panic!("no registry must refuse instead of inventing an empty response");
        };
        assert_eq!(
            error.reason(),
            super::super::AnalysisUnavailableReason::RegistryUnavailable
        );
    }
}
