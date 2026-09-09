//! Server-side utterance segmentation orchestrator.
//!
//! Owns the full CHAT lifecycle for utseg jobs:
//! parse → collect payloads → infer → apply splits → serialize.
//!
//! Python workers receive only `(words, text) → UtsegResponse` via the infer protocol
//! pure Stanza constituency parsing with zero CHAT awareness.

use std::collections::HashMap;

use async_trait::async_trait;
use serde::Serialize;

use crate::api::{ChatText, EngineVersion, LanguageCode3};
use crate::chat_ops::ChatFile;
use crate::types::worker_v2::{
    UtsegAdjacencyPolicyRevisionV2, UtsegBoundaryModelEvidenceV2, UtsegItemResultV2,
};
use crate::worker::artifacts_v2::PreparedArtifactRuntimeV2;
use crate::worker::pool::WorkerPool;
use crate::worker::text_request_v2::{PreparedTextRequestIdsV2, build_utseg_request_v2};
use crate::worker::text_result_v2::parse_utseg_result_v2;
use batchalign_transform::utseg::{
    UtsegBatchItem, UtsegResponse, apply_utseg_results, collect_utseg_payloads,
};

/// Thin adapter matching the legacy `fn(&ChatFile) -> Vec<(usize, Item)>`
/// hook signature. The Wave 5 utseg collector returns the richer
/// [`UtsegPayloadCollection`](batchalign_transform::utseg::UtsegPayloadCollection)
/// struct; this wrapper discards the `not_applicable` outcomes so the
/// existing text-pipeline hooks keep compiling. Surfacing the outcomes
/// through the pipeline is future follow-up work; the data is already
/// typed and available to any caller that calls `collect_utseg_payloads`
/// directly.
fn collect_utseg_batch_items(chat_file: &ChatFile) -> Vec<(usize, UtsegBatchItem)> {
    collect_utseg_payloads(chat_file).batch_items
}
use batchalign_transform::utseg_compute;
use batchalign_transform::validate::ValidityLevel;
use tracing::{info, warn};

use crate::error::ServerError;
use crate::infer_retry::{Cancellation, dispatch_execute_v2_with_retry};
use crate::params::UtsegFallbackPolicy;
use crate::pipeline::PipelineServices;
use crate::pipeline::text_infer::{
    TextBatchHooks, TextPipelineHooks, run_text_batch_pipeline, run_text_pipeline,
};
use crate::text_batch::{
    TextBatchFileInput, TextBatchFileResults, TextBatchOperation, TextBatchWorkflow,
    TextBatchWorkflowRequest, TextPerFileWorkflowRequest,
};

/// Command-specific parameters for the utseg workflow family.
///
/// No `Default`: a caller must state its cancellation posture explicitly
/// (a real token or a named `NotWired` reason), never inherit a silent
/// zero-value stand-in.
#[derive(Debug, Clone, Copy)]
pub(crate) struct UtsegWorkflowParams<'a> {
    /// Operator opt-in to the legacy Stanza constituency-parser
    /// fallback when no language-specific TalkBank BERT utseg model is
    /// configured. Set by the `--utseg-fallback-stanza` CLI flag.
    pub fallback_policy: UtsegFallbackPolicy,
    /// The job's cancellation token, when this dispatch has one.
    pub cancellation: Cancellation<'a>,
}

/// How admitted utterance-boundary decisions enter the local transform.
///
/// Production honors the worker-declared assignments. Controlled offline
/// experiments can instead rederive assignments from the same retained raw
/// boundary evidence under one closed policy revision.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) enum UtsegDecisionPolicy {
    /// Apply the worker's admitted assignments unchanged.
    #[default]
    WorkerDeclared,
    /// Reapply a policy to boundary-model raw actions without model inference.
    ReapplyBoundaryModel(UtsegAdjacencyPolicyRevisionV2),
    /// Reapply an adjacency policy, then suppress only those resulting splits
    /// that would destroy an exact retrace recognized by CHAT cleanup.
    ReapplyBoundaryModelPreservingExactRetraces(UtsegAdjacencyPolicyRevisionV2),
}

/// Which utterance-model passes exist in one transcribe execution.
///
/// A post-CHAT pass cannot exist without an explicit pre-CHAT policy. This
/// makes the historical double pass visible and makes `--no-utseg` genuinely
/// incapable of reaching either model boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TranscribeUtsegExecution {
    Disabled,
    PreChatOnly {
        pre_chat: UtsegDecisionPolicy,
    },
    PreAndPostChat {
        pre_chat: UtsegDecisionPolicy,
        post_chat: UtsegDecisionPolicy,
    },
}

impl TranscribeUtsegExecution {
    pub(crate) fn production(enabled: bool) -> Self {
        if enabled {
            Self::PreAndPostChat {
                pre_chat: UtsegDecisionPolicy::WorkerDeclared,
                post_chat: UtsegDecisionPolicy::WorkerDeclared,
            }
        } else {
            Self::Disabled
        }
    }

    pub(crate) fn pre_chat_policy(self) -> Option<UtsegDecisionPolicy> {
        match self {
            Self::Disabled => None,
            Self::PreChatOnly { pre_chat } | Self::PreAndPostChat { pre_chat, .. } => {
                Some(pre_chat)
            }
        }
    }

    pub(crate) fn post_chat_policy(self) -> Option<UtsegDecisionPolicy> {
        match self {
            Self::Disabled | Self::PreChatOnly { .. } => None,
            Self::PreAndPostChat { post_chat, .. } => Some(post_chat),
        }
    }
}

/// Closed local post-inference policy recorded with every rederived decision.
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum LocalUtsegDecisionPolicyRevision {
    AdjacencyOnlyV1,
    AdjacencyPreserveExactRetracesV1,
}

/// Complete receipt for a locally rederived boundary-model decision.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct LocalUtsegDecisionReceipt {
    revision: LocalUtsegDecisionPolicyRevision,
    worker_adjacency_policy_revision: UtsegAdjacencyPolicyRevisionV2,
    local_adjacency_policy_revision: UtsegAdjacencyPolicyRevisionV2,
    worker_assignments: Vec<usize>,
    suppressed_split_before_word_indices: Vec<usize>,
}

/// Complete post-CHAT utseg request whose evidence destination is mandatory.
///
/// Keeping the sink and filename in the same request prevents callers from
/// invoking the observed path with only half of its retention capability.
pub(crate) struct EvidenceRetainingUtsegRequest<'a> {
    pub(crate) chat_text: ChatText<'a>,
    pub(crate) lang: &'a LanguageCode3,
    pub(crate) services: PipelineServices<'a>,
    pub(crate) fallback_policy: UtsegFallbackPolicy,
    pub(crate) decision_policy: UtsegDecisionPolicy,
    pub(crate) evidence_filename: &'a str,
    pub(crate) evidence_sink: &'a crate::utseg_evidence::UtsegEvidenceSink,
    /// The job's cancellation token, when this dispatch has one.
    pub(crate) cancellation: Cancellation<'a>,
}

/// Typed workflow operation for utseg.
pub(crate) struct UtsegOperation;

/// Trait-oriented workflow wrapper for utseg.
pub(crate) type UtsegWorkflow = TextBatchWorkflow<UtsegOperation>;

#[async_trait]
impl TextBatchOperation for UtsegOperation {
    type Shared<'a>
        = PipelineServices<'a>
    where
        Self: 'a;

    type Params<'a>
        = UtsegWorkflowParams<'a>
    where
        Self: 'a;

    async fn run_single(
        chat_text: ChatText<'_>,
        lang: &LanguageCode3,
        shared: Self::Shared<'_>,
        params: Self::Params<'_>,
    ) -> Result<String, ServerError> {
        run_utseg_impl(
            chat_text.as_ref(),
            lang,
            shared.pool,
            shared.cache,
            shared.engine_version,
            params.fallback_policy.is_allowed(),
            params.cancellation,
        )
        .await
    }

    async fn run_batch(
        files: &[TextBatchFileInput],
        lang: &LanguageCode3,
        shared: Self::Shared<'_>,
        params: Self::Params<'_>,
    ) -> TextBatchFileResults {
        run_utseg_batch_impl(
            files,
            lang,
            shared.pool,
            params.fallback_policy.is_allowed(),
            params.cancellation,
        )
        .await
    }
}

// ---------------------------------------------------------------------------
// Per-file utseg processing
// ---------------------------------------------------------------------------

/// Process a single CHAT file through the utseg pipeline.
///
/// Returns the serialized CHAT text with utterances split as needed.
pub async fn process_utseg(
    chat_text: &str,
    lang: &LanguageCode3,
    pool: &WorkerPool,
    cache: &crate::cache::UtteranceCache,
    engine_version: &EngineVersion,
    allow_stanza_fallback: bool,
    cancellation: Cancellation<'_>,
) -> Result<String, ServerError> {
    UtsegWorkflow::new()
        .run_per_file(TextPerFileWorkflowRequest {
            chat_text: ChatText::from(chat_text),
            lang,
            shared: PipelineServices::new(pool, cache, engine_version),
            params: UtsegWorkflowParams {
                fallback_policy: allow_stanza_fallback.into(),
                cancellation,
            },
        })
        .await
}

/// Process CHAT while durably retaining the exact post-CHAT segmentation
/// evidence requested for a transcribe experiment.
pub(crate) async fn process_utseg_with_evidence(
    request: EvidenceRetainingUtsegRequest<'_>,
) -> Result<String, ServerError> {
    let EvidenceRetainingUtsegRequest {
        chat_text,
        lang,
        services,
        fallback_policy,
        decision_policy,
        evidence_filename,
        evidence_sink,
        cancellation,
    } = request;
    run_utseg_impl_observed(
        chat_text.as_ref(),
        lang,
        services,
        fallback_policy.is_allowed(),
        decision_policy,
        cancellation,
        |requests, predictions| {
            let trace = crate::utseg_evidence::UtsegEvidenceTrace::from_predictions(
                crate::utseg_evidence::UtsegEvidencePhase::PostChat,
                lang.as_ref(),
                services.engine_version.as_ref(),
                requests,
                predictions,
            )
            .map_err(|error| ServerError::Validation(error.to_string()))?;
            evidence_sink
                .write(evidence_filename, &trace)
                .map_err(|error| {
                    ServerError::Persistence(format!(
                        "could not retain requested post-CHAT utseg evidence for {evidence_filename}: {error}"
                    ))
                })?;
            Ok(())
        },
    )
    .await
}

/// Infer utterance-boundary assignments for pretokenized word batches.
///
/// Per-item engine/network/model failures collapse into a single typed
/// ``ServerError::Validation`` carrying the rendered list of failing
/// items (via ``TextWorkflowFileError::item_errors``). Callers that
/// only need a flat success-or-fail signal can rely on this; callers
/// that need per-item attribution (the cross-file pipeline driver)
/// call ``infer_batch`` directly.
pub async fn infer_utseg_assignments(
    pool: &WorkerPool,
    lang: &LanguageCode3,
    items: &[UtsegBatchItem],
    allow_stanza_fallback: bool,
    cancellation: Cancellation<'_>,
) -> Result<Vec<UtsegResponse>, ServerError> {
    infer_utseg_predictions(pool, lang, items, allow_stanza_fallback, cancellation)
        .await
        .map(|predictions| {
            predictions
                .into_iter()
                .map(AdmittedUtsegPrediction::into_response)
                .collect()
        })
}

/// Infer utterance boundaries while retaining the typed source/evidence state.
pub(crate) async fn infer_utseg_predictions(
    pool: &WorkerPool,
    lang: &LanguageCode3,
    items: &[UtsegBatchItem],
    allow_stanza_fallback: bool,
    cancellation: Cancellation<'_>,
) -> Result<Vec<AdmittedUtsegPrediction>, ServerError> {
    infer_utseg_predictions_with_policy(
        pool,
        lang,
        items,
        allow_stanza_fallback,
        UtsegDecisionPolicy::WorkerDeclared,
        cancellation,
    )
    .await
}

/// Infer once, then optionally rederive assignments from retained raw boundary
/// evidence under a closed local policy.
pub(crate) async fn infer_utseg_predictions_with_policy(
    pool: &WorkerPool,
    lang: &LanguageCode3,
    items: &[UtsegBatchItem],
    allow_stanza_fallback: bool,
    decision_policy: UtsegDecisionPolicy,
    cancellation: Cancellation<'_>,
) -> Result<Vec<AdmittedUtsegPrediction>, ServerError> {
    let indexed_items: Vec<(usize, UtsegBatchItem)> = items.iter().cloned().enumerate().collect();
    let item_results = infer_admitted_batch_with_policy(
        pool,
        &indexed_items,
        lang,
        allow_stanza_fallback,
        decision_policy,
        cancellation,
    )
    .await?;
    crate::text_batch::unwrap_per_item_results("utseg", item_results)
        .map_err(|err| ServerError::Validation(err.to_string()))
}

// ---------------------------------------------------------------------------
// Cross-file batch utseg processing
// ---------------------------------------------------------------------------

/// Process multiple CHAT files, pooling payloads from all files into a single
/// `batch_infer` call for maximum throughput.
///
/// Returns `(filename, Ok(output_text) | Err(error_msg))` for each file.
pub(crate) async fn process_utseg_batch(
    files: &[TextBatchFileInput],
    lang: &LanguageCode3,
    pool: &WorkerPool,
    cache: &crate::cache::UtteranceCache,
    engine_version: &EngineVersion,
    allow_stanza_fallback: bool,
    cancellation: Cancellation<'_>,
) -> TextBatchFileResults {
    UtsegWorkflow::new()
        .run_batch_files(TextBatchWorkflowRequest {
            files,
            lang,
            shared: PipelineServices::new(pool, cache, engine_version),
            params: UtsegWorkflowParams {
                fallback_policy: allow_stanza_fallback.into(),
                cancellation,
            },
        })
        .await
}

async fn run_utseg_impl(
    chat_text: &str,
    lang: &LanguageCode3,
    pool: &WorkerPool,
    cache: &crate::cache::UtteranceCache,
    engine_version: &EngineVersion,
    allow_stanza_fallback: bool,
    cancellation: Cancellation<'_>,
) -> Result<String, ServerError> {
    run_utseg_impl_observed(
        chat_text,
        lang,
        PipelineServices::new(pool, cache, engine_version),
        allow_stanza_fallback,
        UtsegDecisionPolicy::WorkerDeclared,
        cancellation,
        |_, _| Ok(()),
    )
    .await
}

async fn run_utseg_impl_observed<Observe>(
    chat_text: &str,
    lang: &LanguageCode3,
    services: PipelineServices<'_>,
    allow_stanza_fallback: bool,
    decision_policy: UtsegDecisionPolicy,
    cancellation: Cancellation<'_>,
    observe: Observe,
) -> Result<String, ServerError>
where
    Observe:
        FnOnce(&[(usize, UtsegBatchItem)], &[AdmittedUtsegPrediction]) -> Result<(), ServerError>,
{
    run_text_pipeline(
        chat_text,
        lang,
        services,
        TextPipelineHooks {
            command: crate::api::ReleasedCommand::Utseg,
            validity: ValidityLevel::StructurallyComplete,
            collect: collect_utseg_batch_items,
            integrate: integrate_admitted_assignments,
            apply: apply_utseg_results,
        },
        // The generic pipeline's `infer` signature doesn't carry
        // command-specific state, so capture the operator opt-in (and
        // now the cancellation) here and bind it onto each
        // `infer_batch` invocation.
        async move |pool, items, lang| {
            infer_admitted_batch_with_policy(
                pool,
                items,
                lang,
                allow_stanza_fallback,
                decision_policy,
                cancellation,
            )
            .await
        },
        observe,
    )
    .await
    // The proof stops here: the single-file entry point is the library/CLI
    // surface and returns text. The gate itself still ran inside
    // `run_text_pipeline`, so a refusal is already a `ServerError`.
    .map(crate::pipeline::post_validate::PostValidated::into_text)
}

async fn run_utseg_batch_impl(
    files: &[TextBatchFileInput],
    lang: &LanguageCode3,
    pool: &WorkerPool,
    allow_stanza_fallback: bool,
    cancellation: Cancellation<'_>,
) -> TextBatchFileResults {
    run_text_batch_pipeline(
        files,
        lang,
        pool,
        TextBatchHooks {
            command: crate::api::ReleasedCommand::Utseg,
            validity: ValidityLevel::StructurallyComplete,
            collect: collect_utseg_batch_items,
            apply: apply_utseg_file,
        },
        async move |pool, items, lang| {
            infer_batch(pool, items, lang, allow_stanza_fallback, cancellation).await
        },
    )
    .await
}

/// Apply utseg responses for one file, skipping items with a
/// length-mismatched assignment vector.
fn apply_utseg_file(
    chat_file: &mut ChatFile,
    items: &[(usize, UtsegBatchItem)],
    responses: &[UtsegResponse],
) {
    let mut assignment_map: HashMap<usize, Vec<usize>> = HashMap::new();
    for ((utt_ordinal, item), resp) in items.iter().zip(responses.iter()) {
        if resp.assignments.len() == item.words.len() {
            assignment_map.insert(*utt_ordinal, resp.assignments.clone());
        } else {
            warn!(
                utterance = utt_ordinal,
                expected = item.words.len(),
                got = resp.assignments.len(),
                "utseg assignment length mismatch, keeping original"
            );
        }
    }
    if !assignment_map.is_empty() {
        apply_utseg_results(chat_file, &assignment_map);
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// A worker result whose mutually exclusive payload and parallel-vector
/// invariants have been checked against the exact dispatched request.
///
/// The variants keep evidence-bearing classifier output distinct from
/// unobserved legacy assignments and Stanza constituency output. Callers must
/// therefore make an explicit choice before discarding provenance.
#[derive(Debug, Clone)]
pub(crate) enum AdmittedUtsegPrediction {
    /// Direct assignments accompanied by per-word boundary-model evidence.
    BoundaryModelWorkerDeclared {
        /// Assignments safe to apply to the request words.
        response: UtsegResponse,
        /// Model identity and evidence parallel to the request words.
        evidence: UtsegBoundaryModelEvidenceV2,
    },
    /// Boundary-model evidence whose applicable assignments were locally
    /// rederived under a fully recorded closed policy.
    BoundaryModelLocallyReapplied {
        response: UtsegResponse,
        evidence: UtsegBoundaryModelEvidenceV2,
        receipt: LocalUtsegDecisionReceipt,
    },
    /// Direct assignments from a worker that did not expose model evidence.
    UnobservedAssignments {
        /// Assignments safe to apply to the request words.
        response: UtsegResponse,
    },
    /// Assignments derived from Stanza constituency trees.
    Constituency {
        /// Assignments safe to apply to the request words.
        response: UtsegResponse,
    },
}

impl AdmittedUtsegPrediction {
    /// Borrow assignments known to be parallel to the dispatched words.
    pub(crate) fn response(&self) -> &UtsegResponse {
        match self {
            Self::BoundaryModelWorkerDeclared {
                response,
                evidence: _,
            }
            | Self::BoundaryModelLocallyReapplied {
                response,
                evidence: _,
                receipt: _,
            }
            | Self::UnobservedAssignments { response }
            | Self::Constituency { response } => response,
        }
    }

    /// Deliberately project an admitted prediction onto the legacy transform
    /// response. Evidence-aware callers retain the enum instead.
    fn into_response(self) -> UtsegResponse {
        match self {
            Self::BoundaryModelWorkerDeclared {
                response,
                evidence: _,
            }
            | Self::BoundaryModelLocallyReapplied {
                response,
                evidence: _,
                receipt: _,
            }
            | Self::UnobservedAssignments { response }
            | Self::Constituency { response } => response,
        }
    }

    fn apply_decision_policy(
        self,
        request: &UtsegBatchItem,
        lang: &LanguageCode3,
        policy: UtsegDecisionPolicy,
    ) -> Result<Self, String> {
        match policy {
            UtsegDecisionPolicy::WorkerDeclared => Ok(self),
            UtsegDecisionPolicy::ReapplyBoundaryModel(policy) => self
                .locally_reapply_boundary_model(
                    request,
                    lang,
                    policy,
                    LocalUtsegDecisionPolicyRevision::AdjacencyOnlyV1,
                ),
            UtsegDecisionPolicy::ReapplyBoundaryModelPreservingExactRetraces(policy) => self
                .locally_reapply_boundary_model(
                    request,
                    lang,
                    policy,
                    LocalUtsegDecisionPolicyRevision::AdjacencyPreserveExactRetracesV1,
                ),
        }
    }

    fn locally_reapply_boundary_model(
        self,
        request: &UtsegBatchItem,
        lang: &LanguageCode3,
        policy: UtsegAdjacencyPolicyRevisionV2,
        revision: LocalUtsegDecisionPolicyRevision,
    ) -> Result<Self, String> {
        match self {
            Self::BoundaryModelWorkerDeclared { response, evidence } => {
                let worker_adjacency_policy_revision = evidence.adjacency_policy_revision;
                let worker_assignments = response.assignments;
                let reapplied = evidence.reapply_adjacency_policy(policy);
                let (evidence, assignments, suppressed_split_before_word_indices) = match revision {
                    LocalUtsegDecisionPolicyRevision::AdjacencyOnlyV1 => {
                        let (evidence, assignments) = reapplied.into_parts();
                        (evidence, assignments, Vec::new())
                    }
                    LocalUtsegDecisionPolicyRevision::AdjacencyPreserveExactRetracesV1 => {
                        let analysis =
                            batchalign_transform::asr_postprocess::analyze_exact_retraces(
                                &request.words,
                                lang.as_ref(),
                            );
                        let protected: Vec<_> = analysis.protected_split_indices().collect();
                        reapplied
                            .protect_splits_before(&protected, &worker_assignments)
                            .map_err(|error| error.to_string())?
                            .into_parts()
                    }
                };
                Ok(Self::BoundaryModelLocallyReapplied {
                    response: UtsegResponse { assignments },
                    evidence,
                    receipt: LocalUtsegDecisionReceipt {
                        revision,
                        worker_adjacency_policy_revision,
                        local_adjacency_policy_revision: policy,
                        worker_assignments,
                        suppressed_split_before_word_indices,
                    },
                })
            }
            Self::BoundaryModelLocallyReapplied { .. } => {
                Err("cannot apply a second local utterance-boundary policy".into())
            }
            Self::UnobservedAssignments { response: _ } => {
                Err("cannot reapply an utterance-boundary policy to unobserved assignments".into())
            }
            Self::Constituency { response: _ } => {
                Err("cannot reapply an utterance-boundary policy to constituency output".into())
            }
        }
    }
}

/// Validate one raw worker item before it can become an applicable response.
fn admit_worker_item(
    request: &UtsegBatchItem,
    result: &UtsegItemResultV2,
) -> Result<AdmittedUtsegPrediction, String> {
    if let Some(error) = &result.error {
        if result.assignments.is_some()
            || result.trees.is_some()
            || result.boundary_model_evidence.is_some()
        {
            return Err("utseg V2 returned an error together with a success payload".to_owned());
        }
        return Err(error.clone());
    }

    let response = match (&result.assignments, &result.trees) {
        (Some(_), Some(_)) => {
            return Err(
                "utseg V2 returned both direct assignments and constituency trees".to_owned(),
            );
        }
        (None, None) => {
            if result.boundary_model_evidence.is_some() {
                return Err(
                    "utseg V2 returned boundary evidence without direct assignments".to_owned(),
                );
            }
            return Err("utseg V2 returned no assignments, no trees, and no error".to_owned());
        }
        (Some(assignments), None) => UtsegResponse {
            assignments: assignments.clone(),
        },
        (None, Some(trees)) => UtsegResponse {
            assignments: utseg_compute::compute_assignments(trees, request.words.len()),
        },
    };

    if response.assignments.len() != request.words.len() {
        return Err(format!(
            "utseg V2 returned {} assignments for {} request words",
            response.assignments.len(),
            request.words.len()
        ));
    }

    match (&result.assignments, &result.boundary_model_evidence) {
        (Some(_), Some(evidence)) => {
            if evidence.model_id.is_empty() {
                return Err("utseg V2 boundary evidence has an empty model id".to_owned());
            }
            if evidence.model_revision.as_deref() == Some("") {
                return Err("utseg V2 boundary evidence has an empty model revision".to_owned());
            }
            if evidence.word_evidence.len() != request.words.len() {
                return Err(format!(
                    "utseg V2 returned {} boundary evidence states for {} request words",
                    evidence.word_evidence.len(),
                    request.words.len()
                ));
            }
            evidence
                .validate_assignments(&response.assignments)
                .map_err(|error| format!("utseg V2 boundary evidence {error}"))?;
            Ok(AdmittedUtsegPrediction::BoundaryModelWorkerDeclared {
                response,
                evidence: evidence.clone(),
            })
        }
        (Some(_), None) => Ok(AdmittedUtsegPrediction::UnobservedAssignments { response }),
        (None, None) => Ok(AdmittedUtsegPrediction::Constituency { response }),
        (None, Some(_)) => {
            Err("utseg V2 returned boundary evidence with constituency trees".to_owned())
        }
    }
}

/// Send batch items to a worker for constituency inference via batched
/// `execute_v2`.
///
/// `allow_stanza_fallback` propagates the operator opt-in
/// (`--utseg-fallback-stanza`) to the worker so it can engage the
/// legacy Stanza constituency-parser fallback when no
/// language-specific BERT utseg model is configured.
async fn infer_batch(
    pool: &WorkerPool,
    items: &[(usize, UtsegBatchItem)],
    lang: &LanguageCode3,
    allow_stanza_fallback: bool,
    cancellation: Cancellation<'_>,
) -> Result<Vec<Result<UtsegResponse, String>>, ServerError> {
    Ok(
        infer_admitted_batch(pool, items, lang, allow_stanza_fallback, cancellation)
            .await?
            .into_iter()
            .map(|result| result.map(AdmittedUtsegPrediction::into_response))
            .collect(),
    )
}

/// Dispatch and admit a batch without erasing its inference-source state.
async fn infer_admitted_batch(
    pool: &WorkerPool,
    items: &[(usize, UtsegBatchItem)],
    lang: &LanguageCode3,
    allow_stanza_fallback: bool,
    cancellation: Cancellation<'_>,
) -> Result<Vec<Result<AdmittedUtsegPrediction, String>>, ServerError> {
    infer_admitted_batch_with_policy(
        pool,
        items,
        lang,
        allow_stanza_fallback,
        UtsegDecisionPolicy::WorkerDeclared,
        cancellation,
    )
    .await
}

async fn infer_admitted_batch_with_policy(
    pool: &WorkerPool,
    items: &[(usize, UtsegBatchItem)],
    lang: &LanguageCode3,
    allow_stanza_fallback: bool,
    decision_policy: UtsegDecisionPolicy,
    cancellation: Cancellation<'_>,
) -> Result<Vec<Result<AdmittedUtsegPrediction, String>>, ServerError> {
    let payload_items: Vec<_> = items.iter().map(|(_, item)| item.clone()).collect();
    let artifacts = PreparedArtifactRuntimeV2::new("utseg_v2").map_err(|error| {
        ServerError::Validation(format!(
            "failed to create utseg V2 artifact runtime: {error}"
        ))
    })?;
    let request_ids = PreparedTextRequestIdsV2::for_task("utseg");
    let request = build_utseg_request_v2(
        artifacts.store(),
        &request_ids,
        lang,
        &payload_items,
        allow_stanza_fallback,
    )
    .map_err(|error| {
        ServerError::Validation(format!("failed to build utseg V2 worker request: {error}"))
    })?;

    info!(
        num_items = items.len(),
        lang = %lang,
        "Dispatching utseg execute_v2 batch"
    );

    let response = dispatch_execute_v2_with_retry(pool, lang, &request, cancellation).await?;
    let result = parse_utseg_result_v2(&response)
        .map_err(|error| ServerError::Validation(format!("invalid utseg V2 result: {error}")))?;
    if result.items.len() != items.len() {
        return Err(ServerError::Validation(format!(
            "utseg V2 returned {} items for {} requests",
            result.items.len(),
            items.len()
        )));
    }

    let mut admitted = Vec::with_capacity(result.items.len());
    for (i, item_result) in result.items.iter().enumerate() {
        admitted.push(
            admit_worker_item(&items[i].1, item_result).and_then(|prediction| {
                prediction.apply_decision_policy(&items[i].1, lang, decision_policy)
            }),
        );
    }

    Ok(admitted)
}

fn integrate_admitted_assignments(
    assignment_map: &mut HashMap<usize, Vec<usize>>,
    misses: &[(usize, UtsegBatchItem)],
    predictions: &[AdmittedUtsegPrediction],
) {
    for ((utt_ordinal, _item), prediction) in misses.iter().zip(predictions.iter()) {
        assignment_map.insert(*utt_ordinal, prediction.response().assignments.clone());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::worker_v2::{
        BoundaryProbabilityMicrosV2, UtsegAdjacencyPolicyRevisionV2, UtsegBoundaryActionV2,
        UtsegBoundaryModelEvidenceV2, UtsegItemResultV2, UtsegNormalizationRevisionV2,
        UtsegWordBoundaryEvidenceV2,
    };

    #[test]
    fn disabled_transcribe_utseg_has_no_reachable_model_pass() {
        let execution = TranscribeUtsegExecution::production(false);
        assert_eq!(execution.pre_chat_policy(), None);
        assert_eq!(execution.post_chat_policy(), None);
    }

    #[test]
    fn pre_chat_only_execution_cannot_reach_the_post_chat_policy() {
        let execution = TranscribeUtsegExecution::PreChatOnly {
            pre_chat: UtsegDecisionPolicy::WorkerDeclared,
        };
        assert_eq!(
            execution.pre_chat_policy(),
            Some(UtsegDecisionPolicy::WorkerDeclared)
        );
        assert_eq!(execution.post_chat_policy(), None);
    }

    fn two_word_request() -> UtsegBatchItem {
        UtsegBatchItem {
            words: vec!["one".to_owned(), "two".to_owned()],
            text: "one two".to_owned(),
        }
    }

    fn boundary_result(evidence_words: usize) -> UtsegItemResultV2 {
        UtsegItemResultV2 {
            assignments: Some(vec![0, 1]),
            trees: None,
            boundary_model_evidence: Some(UtsegBoundaryModelEvidenceV2 {
                model_id: "talkbank/utterance-boundary".to_owned(),
                model_revision: Some("revision-1".to_owned()),
                normalization_revision: UtsegNormalizationRevisionV2::LowerStripAsciiPunctuationV1,
                adjacency_policy_revision:
                    UtsegAdjacencyPolicyRevisionV2::SuppressEarlierAdjacentNonordinaryV1,
                word_evidence: (0..evidence_words)
                    .map(|index| {
                        if index == 0 {
                            UtsegWordBoundaryEvidenceV2::Classified {
                                raw_action: UtsegBoundaryActionV2::PeriodBoundary,
                                applied_action: UtsegBoundaryActionV2::PeriodBoundary,
                                boundary_probability_micros: BoundaryProbabilityMicrosV2::try_from(
                                    900_000,
                                )
                                .expect("valid fixture probability"),
                            }
                        } else {
                            UtsegWordBoundaryEvidenceV2::Classified {
                                raw_action: UtsegBoundaryActionV2::Ordinary,
                                applied_action: UtsegBoundaryActionV2::Ordinary,
                                boundary_probability_micros: BoundaryProbabilityMicrosV2::try_from(
                                    10_000,
                                )
                                .expect("valid fixture probability"),
                            }
                        }
                    })
                    .collect(),
            }),
            error: None,
        }
    }

    #[test]
    fn admits_boundary_prediction_only_when_all_parallel_vectors_align() {
        let admitted = admit_worker_item(&two_word_request(), &boundary_result(2))
            .expect("aligned boundary result should be admitted");

        let AdmittedUtsegPrediction::BoundaryModelWorkerDeclared { response, evidence } = admitted
        else {
            panic!("direct classifier result should retain its boundary-model evidence");
        };
        assert_eq!(response.assignments, vec![0, 1]);
        assert_eq!(evidence.word_evidence.len(), 2);
        assert_eq!(evidence.model_id, "talkbank/utterance-boundary");
    }

    #[test]
    fn refuses_boundary_evidence_that_is_not_parallel_to_request_words() {
        let error = admit_worker_item(&two_word_request(), &boundary_result(1))
            .expect_err("misaligned evidence must never become an applicable response");

        assert!(error.contains("boundary evidence"));
        assert!(error.contains("2 request words"));
        assert!(error.contains("1 boundary evidence states"));
    }

    #[test]
    fn refuses_assignments_that_disagree_with_applied_boundary_evidence() {
        let mut result = boundary_result(2);
        result.assignments = Some(vec![0, 0]);

        let error = admit_worker_item(&two_word_request(), &result)
            .expect_err("evidence and assignments must describe one decision");

        assert!(error.contains("assignments disagree"));
    }

    #[test]
    fn refuses_applied_actions_that_disagree_with_declared_policy() {
        let mut result = boundary_result(2);
        let evidence = result
            .boundary_model_evidence
            .as_mut()
            .expect("fixture has boundary evidence");
        evidence.word_evidence[1] = UtsegWordBoundaryEvidenceV2::Classified {
            raw_action: UtsegBoundaryActionV2::CapitalizedOnset,
            applied_action: UtsegBoundaryActionV2::CapitalizedOnset,
            boundary_probability_micros: BoundaryProbabilityMicrosV2::try_from(10_000)
                .expect("valid fixture probability"),
        };

        let error = admit_worker_item(&two_word_request(), &result)
            .expect_err("declared policy must explain applied actions");

        assert!(error.contains("applied action disagrees"));
    }

    #[test]
    fn typed_candidate_policy_rederives_assignments_from_raw_evidence() {
        let mut result = boundary_result(2);
        let evidence = result
            .boundary_model_evidence
            .as_mut()
            .expect("fixture has evidence");
        evidence.word_evidence[0] = UtsegWordBoundaryEvidenceV2::Classified {
            raw_action: UtsegBoundaryActionV2::PeriodBoundary,
            applied_action: UtsegBoundaryActionV2::Ordinary,
            boundary_probability_micros: BoundaryProbabilityMicrosV2::try_from(900_000)
                .expect("probability"),
        };
        evidence.word_evidence[1] = UtsegWordBoundaryEvidenceV2::Classified {
            raw_action: UtsegBoundaryActionV2::CapitalizedOnset,
            applied_action: UtsegBoundaryActionV2::CapitalizedOnset,
            boundary_probability_micros: BoundaryProbabilityMicrosV2::try_from(800_000)
                .expect("probability"),
        };
        result.assignments = Some(vec![0, 0]);
        let request = two_word_request();
        let admitted = admit_worker_item(&request, &result)
            .expect("baseline evidence")
            .apply_decision_policy(
                &request,
                &LanguageCode3::eng(),
                UtsegDecisionPolicy::ReapplyBoundaryModel(
                    UtsegAdjacencyPolicyRevisionV2::SuppressEarlierAdjacentBoundariesV1,
                ),
            )
            .expect("candidate replay");

        assert_eq!(admitted.response().assignments, vec![0, 1]);
    }

    #[test]
    fn candidate_policy_refuses_constituency_output_without_raw_actions() {
        let prediction = AdmittedUtsegPrediction::Constituency {
            response: UtsegResponse {
                assignments: vec![0, 0],
            },
        };
        assert!(
            prediction
                .apply_decision_policy(
                    &two_word_request(),
                    &LanguageCode3::eng(),
                    UtsegDecisionPolicy::ReapplyBoundaryModel(
                        UtsegAdjacencyPolicyRevisionV2::SuppressEarlierAdjacentBoundariesV1,
                    ),
                )
                .is_err()
        );
    }

    #[test]
    fn exact_retrace_guard_suppresses_candidate_splits_inside_both_copies() {
        let words = [
            "How", "can", "I", "take", "it", "off", "blur", "can", "I", "take", "it", "off", "blur",
        ];
        let request = UtsegBatchItem {
            words: words.iter().map(|word| (*word).to_owned()).collect(),
            text: words.join(" "),
        };
        let probability = BoundaryProbabilityMicrosV2::try_from(900_000).expect("probability");
        let mut evidence_words = Vec::new();
        for index in 0..words.len() {
            let (raw_action, applied_action) = match index {
                5 | 11 => (
                    UtsegBoundaryActionV2::PeriodBoundary,
                    UtsegBoundaryActionV2::Ordinary,
                ),
                6 | 12 => (
                    UtsegBoundaryActionV2::CapitalizedOnset,
                    UtsegBoundaryActionV2::CapitalizedOnset,
                ),
                _ => (
                    UtsegBoundaryActionV2::Ordinary,
                    UtsegBoundaryActionV2::Ordinary,
                ),
            };
            evidence_words.push(UtsegWordBoundaryEvidenceV2::Classified {
                raw_action,
                applied_action,
                boundary_probability_micros: probability,
            });
        }
        let result = UtsegItemResultV2 {
            assignments: Some(vec![0; words.len()]),
            trees: None,
            boundary_model_evidence: Some(UtsegBoundaryModelEvidenceV2 {
                model_id: "model".into(),
                model_revision: Some("revision".into()),
                normalization_revision: UtsegNormalizationRevisionV2::LowerStripAsciiPunctuationV1,
                adjacency_policy_revision:
                    UtsegAdjacencyPolicyRevisionV2::SuppressEarlierAdjacentNonordinaryV1,
                word_evidence: evidence_words,
            }),
            error: None,
        };

        let admitted = admit_worker_item(&request, &result)
            .expect("worker evidence")
            .apply_decision_policy(
                &request,
                &LanguageCode3::eng(),
                UtsegDecisionPolicy::ReapplyBoundaryModelPreservingExactRetraces(
                    UtsegAdjacencyPolicyRevisionV2::SuppressEarlierAdjacentBoundariesV1,
                ),
            )
            .expect("guarded local replay");

        let AdmittedUtsegPrediction::BoundaryModelLocallyReapplied {
            response, receipt, ..
        } = admitted
        else {
            panic!("guarded replay must retain its local receipt")
        };
        assert_eq!(response.assignments, vec![0; words.len()]);
        assert_eq!(receipt.suppressed_split_before_word_indices, vec![6, 12]);
    }
}
