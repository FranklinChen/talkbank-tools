//! Server-side utterance segmentation orchestrator.
//!
//! Owns the full CHAT lifecycle for utseg jobs:
//! parse → collect payloads → infer → apply splits → serialize.
//!
//! Python workers receive only `(words, text) → UtsegResponse` via the infer protocol
//! pure Stanza constituency parsing with zero CHAT awareness.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::api::LanguageCode3;
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
pub(crate) fn collect_utseg_batch_items(chat_file: &ChatFile) -> Vec<(usize, UtsegBatchItem)> {
    collect_utseg_payloads(chat_file).batch_items
}
use batchalign_transform::utseg_compute;
use tracing::info;

use crate::error::ServerError;
use crate::infer_retry::{Cancellation, dispatch_execute_v2_with_retry};
use crate::params::UtsegFallbackPolicy;
use crate::pipeline::PipelineServices;
use crate::pipeline::text_infer::{TextBatchHooks, TextPipelineHooks, run_text_batch_pipeline};
use crate::text_batch::{EngineItemFailure, ItemFailure, TextBatchFileInput, TextBatchFileResults};

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
///
/// Deserializable because a retained evidence artifact records it and the
/// replay reads it back (`crate::utseg_evidence::AdmittedUtsegEvidence`).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum LocalUtsegDecisionPolicyRevision {
    AdjacencyOnlyV1,
    AdjacencyPreserveExactRetracesV1,
}

/// Complete receipt for a locally rederived boundary-model decision.
///
/// A receipt claims that one closed local policy, applied to these worker
/// assignments, produced the applicable assignments it travels with. Because it
/// can also be read back from an artifact, that claim is checked wherever a
/// receipt joins a prediction: see [`Self::check_explains`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct LocalUtsegDecisionReceipt {
    revision: LocalUtsegDecisionPolicyRevision,
    worker_adjacency_policy_revision: UtsegAdjacencyPolicyRevisionV2,
    local_adjacency_policy_revision: UtsegAdjacencyPolicyRevisionV2,
    worker_assignments: Vec<usize>,
    suppressed_split_before_word_indices: Vec<usize>,
}

impl LocalUtsegDecisionReceipt {
    /// Prove this receipt explains `evidence` and `response`.
    ///
    /// Recomputed, never trusted, and through the same owner that performed the
    /// original reapplication: the worker's declared policy must reproduce the
    /// worker assignments the receipt records, the evidence must declare the
    /// local policy the receipt names (reapplication stamps it there), and
    /// replaying that local policy must reproduce both the applicable
    /// assignments and the exact suppressions claimed. A receipt describing some
    /// other decision therefore cannot travel with an admitted prediction.
    fn check_explains(
        &self,
        evidence: &UtsegBoundaryModelEvidenceV2,
        response: &UtsegResponse,
    ) -> Result<(), String> {
        if evidence.adjacency_policy_revision != self.local_adjacency_policy_revision {
            return Err(format!(
                "local utterance-boundary receipt names the {:?} adjacency policy, but its \
                 evidence declares {:?}",
                self.local_adjacency_policy_revision, evidence.adjacency_policy_revision
            ));
        }
        if self.worker_assignments.len() != evidence.word_evidence.len() {
            return Err(format!(
                "local utterance-boundary receipt records {} worker assignments for {} \
                 evidence words",
                self.worker_assignments.len(),
                evidence.word_evidence.len()
            ));
        }

        // Reapplication recomputes applied actions from the RAW actions, which
        // it never changes, so replaying the worker's own policy must give the
        // worker's own assignments back.
        let (_, worker_side) = evidence
            .reapply_adjacency_policy(self.worker_adjacency_policy_revision)
            .into_parts();
        if worker_side != self.worker_assignments {
            return Err(
                "local utterance-boundary receipt records worker assignments that the worker's \
                 own declared policy does not produce"
                    .to_owned(),
            );
        }

        let reapplied = evidence.reapply_adjacency_policy(self.local_adjacency_policy_revision);
        let (applicable, suppressed) = match self.revision {
            LocalUtsegDecisionPolicyRevision::AdjacencyOnlyV1 => {
                let (_, assignments) = reapplied.into_parts();
                (assignments, Vec::new())
            }
            LocalUtsegDecisionPolicyRevision::AdjacencyPreserveExactRetracesV1 => {
                let (_, assignments, suppressed) = reapplied
                    .protect_splits_before(
                        &self.suppressed_split_before_word_indices,
                        &self.worker_assignments,
                    )
                    .map_err(|error| error.to_string())?
                    .into_parts();
                (assignments, suppressed)
            }
        };
        if applicable != response.assignments {
            return Err(
                "local utterance-boundary receipt does not reproduce the assignments it \
                 accompanies"
                    .to_owned(),
            );
        }
        if suppressed != self.suppressed_split_before_word_indices {
            return Err(
                "local utterance-boundary receipt claims suppressions its own policy does not make"
                    .to_owned(),
            );
        }
        Ok(())
    }
}

/// Complete post-CHAT utseg request whose evidence destination is mandatory.
///
/// Keeping the sink and filename in the same request prevents callers from
/// invoking the observed path with only half of its retention capability.
///
/// `Document` is what is segmented: an admitted document
/// ([`crate::pipeline::post_validate::PostValidated`]), or a diagnosed one
/// whose findings are confined to some utterances
/// ([`crate::pipeline::post_validate::LocalizedDiagnosis`]), whose other
/// utterances are segmented and whose held-out ones are left as they are.
pub(crate) struct EvidenceRetainingUtsegRequest<'a, Document> {
    /// The document to segment.
    pub(crate) document: Document,
    pub(crate) lang: &'a LanguageCode3,
    pub(crate) services: PipelineServices<'a>,
    pub(crate) fallback_policy: UtsegFallbackPolicy,
    pub(crate) decision_policy: UtsegDecisionPolicy,
    pub(crate) evidence_filename: &'a str,
    pub(crate) evidence_sink: &'a crate::utseg_evidence::UtsegEvidenceSink,
    /// The job's cancellation token, when this dispatch has one.
    pub(crate) cancellation: Cancellation<'a>,
}

// ---------------------------------------------------------------------------
// Per-file utseg processing
// ---------------------------------------------------------------------------
//
// The single-file entry point and the workflow trait implementation that
// reached it are gone: every utseg job runs through the batch path (the
// dispatcher hands the gateway one file per call for durability), and
// transcribe reaches the per-file pipeline directly through
// `process_utseg_with_evidence`, which is the only caller that needs it.

/// Process CHAT while durably retaining the exact post-CHAT segmentation
/// evidence requested for a transcribe experiment.
pub(crate) async fn process_utseg_with_evidence(
    request: EvidenceRetainingUtsegRequest<'_, crate::pipeline::post_validate::PostValidated>,
) -> Result<crate::pipeline::post_validate::PostValidated, ServerError> {
    let EvidenceRetainingUtsegRequest {
        document,
        lang,
        services,
        fallback_policy,
        decision_policy,
        evidence_filename,
        evidence_sink,
        cancellation,
    } = request;
    run_utseg_impl_observed(
        document,
        lang,
        services,
        fallback_policy.is_allowed(),
        decision_policy,
        cancellation,
        post_chat_evidence_observer(lang, evidence_filename, evidence_sink),
    )
    .await
}

/// Segment every utterance of a diagnosed transcript except the ones its
/// findings are confined to, retaining the evidence exactly as for an
/// admitted one. The result is judged afresh by the producer transition.
pub(crate) async fn process_localized_utseg_with_evidence(
    request: EvidenceRetainingUtsegRequest<'_, crate::pipeline::post_validate::LocalizedDiagnosis>,
) -> Result<crate::pipeline::post_validate::ProducedOutput, ServerError> {
    let EvidenceRetainingUtsegRequest {
        document,
        lang,
        services,
        fallback_policy,
        decision_policy,
        evidence_filename,
        evidence_sink,
        cancellation,
    } = request;
    let allow_stanza_fallback = fallback_policy.is_allowed();
    crate::pipeline::text_infer::run_localized_text_pipeline(
        document,
        lang,
        services,
        utseg_hooks(),
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
        post_chat_evidence_observer(lang, evidence_filename, evidence_sink),
    )
    .await
}

/// Retain one file's post-CHAT segmentation evidence where the request said.
fn post_chat_evidence_observer<'a>(
    lang: &'a LanguageCode3,
    evidence_filename: &'a str,
    evidence_sink: &'a crate::utseg_evidence::UtsegEvidenceSink,
) -> impl FnOnce(&[(usize, UtsegBatchItem)], &[AdmittedUtsegPrediction]) -> Result<(), ServerError> + 'a
{
    move |requests, predictions| {
        let trace = crate::utseg_evidence::UtsegEvidenceTrace::from_predictions(
            crate::utseg_evidence::UtsegEvidencePhase::PostChat,
            lang.as_ref(),
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
    }
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
    allow_stanza_fallback: bool,
    cancellation: Cancellation<'_>,
) -> TextBatchFileResults {
    run_utseg_batch_impl(files, lang, pool, allow_stanza_fallback, cancellation).await
}

/// The single-file text-pipeline hooks for utterance segmentation.
fn utseg_hooks() -> TextPipelineHooks<UtsegBatchItem, Vec<usize>, AdmittedUtsegPrediction> {
    TextPipelineHooks {
        command: crate::api::ReleasedCommand::Utseg,
        collect: collect_utseg_batch_items,
        integrate: integrate_admitted_assignments,
        apply: |file, assignments| {
            apply_utseg_document(file, assignments).map_err(ServerError::from)
        },
        provenance: crate::provenance::utseg_provenance,
    }
}

async fn run_utseg_impl_observed<Observe>(
    document: crate::pipeline::post_validate::PostValidated,
    lang: &LanguageCode3,
    services: PipelineServices<'_>,
    allow_stanza_fallback: bool,
    decision_policy: UtsegDecisionPolicy,
    cancellation: Cancellation<'_>,
    observe: Observe,
) -> Result<crate::pipeline::post_validate::PostValidated, ServerError>
where
    Observe:
        FnOnce(&[(usize, UtsegBatchItem)], &[AdmittedUtsegPrediction]) -> Result<(), ServerError>,
{
    crate::pipeline::text_infer::run_admitted_text_pipeline(
        document,
        lang,
        services,
        utseg_hooks(),
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
            collect: collect_utseg_batch_items,
            apply: apply_utseg_predictions,
            provenance: crate::provenance::utseg_provenance,
        },
        // The batch keeps the admitted predictions rather than projecting them
        // to bare responses: the evidence they carry is what names the
        // boundary model in this file's stamp.
        async move |pool, items, lang| {
            infer_admitted_batch(pool, items, lang, allow_stanza_fallback, cancellation).await
        },
    )
    .await
}

/// Apply one file's admitted predictions.
///
/// There is no length check here, and no "keeping original" branch: admission
/// (`admit_worker_item`) refuses any prediction whose assignments are not
/// parallel to the request words, so a mismatched vector has no route to this
/// point. The branch that used to warn and drop the utterance was unreachable,
/// which meant a defect it claimed to handle would have shown up as silently
/// unsegmented output rather than as the failure admission already produces.
fn apply_utseg_predictions(
    chat_file: &mut ChatFile,
    items: &[(usize, UtsegBatchItem)],
    predictions: &[AdmittedUtsegPrediction],
) -> Result<(), ServerError> {
    let mut assignment_map: HashMap<usize, Vec<usize>> = HashMap::new();
    integrate_admitted_assignments(&mut assignment_map, items, predictions);
    if !assignment_map.is_empty() {
        apply_utseg_document(chat_file, &assignment_map)?;
    }
    Ok(())
}

pub(crate) fn apply_utseg_document(
    chat_file: &mut ChatFile,
    assignment_map: &HashMap<usize, Vec<usize>>,
) -> Result<(), batchalign_transform::utseg::UtsegApplyRefusal> {
    let losses = apply_utseg_results(chat_file, assignment_map)?;
    if !losses.is_empty() {
        let descriptions: Vec<_> = losses
            .iter()
            .map(|loss| {
                format!(
                    "input utterance {} %{} ({:?})",
                    loss.utterance_ordinal() + 1,
                    loss.tier_kind(),
                    loss.reason(),
                )
            })
            .collect();
        let text = format!(
            "Utterance segmentation invalidated dependent tiers: {}. Regenerate analysis if required.",
            descriptions.join("; "),
        );
        let insert_pos = crate::provenance::insert_pos_after_constant_headers(chat_file);
        chat_file.lines.insert(
            insert_pos,
            talkbank_model::model::Line::header(talkbank_model::model::Header::Comment {
                content: crate::chat_ops::BulletContent::from_text(text),
            }),
        );
    }
    Ok(())
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

    /// Restate an admitted boundary-model prediction as one whose applicable
    /// assignments were rederived locally, from the receipt retained with it.
    ///
    /// Consumes the admitted value rather than taking its parts, so a receipt
    /// cannot be attached to assignments and evidence that were never admitted,
    /// and the receipt must first prove it explains them
    /// ([`LocalUtsegDecisionReceipt::check_explains`]). A receipt can be
    /// deserialized from an artifact, so this transition treats one as a claim
    /// to check rather than a label to copy.
    pub(crate) fn with_local_decision(
        self,
        receipt: LocalUtsegDecisionReceipt,
    ) -> Result<Self, String> {
        match self {
            Self::BoundaryModelWorkerDeclared { response, evidence } => {
                receipt.check_explains(&evidence, &response)?;
                Ok(Self::BoundaryModelLocallyReapplied {
                    response,
                    evidence,
                    receipt,
                })
            }
            Self::BoundaryModelLocallyReapplied { .. } => {
                Err("cannot record a second local utterance-boundary decision".to_owned())
            }
            Self::UnobservedAssignments { .. } | Self::Constituency { .. } => Err(
                "a local utterance-boundary decision requires boundary-model evidence".to_owned(),
            ),
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

/// What produced one set of assignments, before they have been admitted.
///
/// The single shape both routes into [`admit_prediction`] narrow to: a live
/// worker result (classified by [`admit_worker_item`]) and a retained evidence
/// artifact (classified by [`crate::utseg_evidence::AdmittedUtsegEvidence`]).
/// One shape means one copy of the parallel-vector and consistency checks, so
/// evidence read back from disk is admitted exactly as the worker's own answer
/// was rather than trusted because it is on disk.
pub(crate) enum UtsegPredictionOrigin<'a> {
    /// A boundary model, with its per-word evidence.
    BoundaryModel(&'a UtsegBoundaryModelEvidenceV2),
    /// A worker that returned assignments without naming their source.
    UnnamedWorker,
    /// Stanza constituency trees, already projected to assignments.
    Constituency,
}

/// Admit `assignments` as applicable to `request`, given what produced them.
///
/// The only constructor of an [`AdmittedUtsegPrediction`]. Everything the type
/// promises is established here: one group per request word, boundary evidence
/// parallel to those words, and applied actions and assignments that agree with
/// the policy the evidence declares. A caller cannot reach the type without
/// passing through this function, so there is no route that skips a check.
pub(crate) fn admit_prediction(
    request: &UtsegBatchItem,
    assignments: Vec<usize>,
    origin: UtsegPredictionOrigin<'_>,
) -> Result<AdmittedUtsegPrediction, String> {
    if assignments.len() != request.words.len() {
        return Err(format!(
            "utseg V2 returned {} assignments for {} request words",
            assignments.len(),
            request.words.len()
        ));
    }
    let response = UtsegResponse { assignments };

    match origin {
        UtsegPredictionOrigin::BoundaryModel(evidence) => {
            if evidence.model_id.is_empty() {
                return Err("utseg V2 boundary evidence has an empty model id".to_owned());
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
        UtsegPredictionOrigin::UnnamedWorker => {
            Ok(AdmittedUtsegPrediction::UnobservedAssignments { response })
        }
        UtsegPredictionOrigin::Constituency => {
            Ok(AdmittedUtsegPrediction::Constituency { response })
        }
    }
}

/// Validate one worker item before it can become an applicable response.
///
/// The item is one of the worker's real states, so each maps to one
/// [`UtsegPredictionOrigin`] (or its failure) and [`admit_prediction`] owns
/// every check that follows.
fn admit_worker_item(
    request: &UtsegBatchItem,
    result: &UtsegItemResultV2,
) -> Result<AdmittedUtsegPrediction, String> {
    match result {
        UtsegItemResultV2::Failed { error } => Err(error.clone()),
        UtsegItemResultV2::BoundaryModel {
            assignments,
            boundary_model_evidence,
        } => admit_prediction(
            request,
            assignments.clone(),
            UtsegPredictionOrigin::BoundaryModel(boundary_model_evidence),
        ),
        UtsegItemResultV2::Unattributed { assignments } => admit_prediction(
            request,
            assignments.clone(),
            UtsegPredictionOrigin::UnnamedWorker,
        ),
        // A parse with no tree, or a tree that does not read, is the item's
        // failure: segmenting by what is left would read as one utterance.
        UtsegItemResultV2::Constituency { trees } => admit_prediction(
            request,
            utseg_compute::ConstituencyParse::read(trees)
                .map_err(|error| format!("utseg constituency parse {error}"))?
                .assignments(request.words.len()),
            UtsegPredictionOrigin::Constituency,
        ),
    }
}

/// Dispatch and admit a batch without erasing its inference-source state.
///
/// `allow_stanza_fallback` propagates the operator opt-in
/// (`--utseg-fallback-stanza`) to the worker so it can engage the
/// legacy Stanza constituency-parser fallback when no
/// language-specific BERT utseg model is configured.
async fn infer_admitted_batch(
    pool: &WorkerPool,
    items: &[(usize, UtsegBatchItem)],
    lang: &LanguageCode3,
    allow_stanza_fallback: bool,
    cancellation: Cancellation<'_>,
) -> Result<Vec<Result<AdmittedUtsegPrediction, EngineItemFailure>>, ServerError> {
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
) -> Result<Vec<Result<AdmittedUtsegPrediction, EngineItemFailure>>, ServerError> {
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
            admit_worker_item(&items[i].1, item_result)
                .and_then(|prediction| {
                    prediction.apply_decision_policy(&items[i].1, lang, decision_policy)
                })
                // Everything refused here is the worker's own answer about
                // this item: a refusal will repeat if the same result comes
                // back, which is what the terminal class means.
                .map_err(ItemFailure::EngineReported),
        );
    }

    Ok(admitted)
}

pub(crate) fn integrate_admitted_assignments(
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
        BoundaryProbabilityMicrosV2, HubCommitV2, UtsegAdjacencyPolicyRevisionV2,
        UtsegBoundaryActionV2, UtsegBoundaryModelEvidenceV2, UtsegItemResultV2,
        UtsegNormalizationRevisionV2, UtsegWordBoundaryEvidenceV2,
    };

    /// A commit-shaped revision for fixtures; the type admits nothing else.
    fn test_commit() -> HubCommitV2 {
        HubCommitV2::try_from("0123456789abcdef0123456789abcdef01234567")
            .expect("valid fixture commit")
    }

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
        UtsegItemResultV2::BoundaryModel {
            assignments: vec![0, 1],
            boundary_model_evidence: UtsegBoundaryModelEvidenceV2 {
                model_id: "talkbank/utterance-boundary".to_owned(),
                model_revision: test_commit(),
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
            },
        }
    }

    /// The assignments and evidence of a boundary-model fixture, for a test
    /// to change.
    fn boundary_parts(
        result: &mut UtsegItemResultV2,
    ) -> (&mut Vec<usize>, &mut UtsegBoundaryModelEvidenceV2) {
        match result {
            UtsegItemResultV2::BoundaryModel {
                assignments,
                boundary_model_evidence,
            } => (assignments, boundary_model_evidence),
            UtsegItemResultV2::Unattributed { .. }
            | UtsegItemResultV2::Constituency { .. }
            | UtsegItemResultV2::Failed { .. } => panic!("a boundary-model fixture"),
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

    /// A constituency item with no tree, or with a tree that does not read,
    /// is the item's failure: it used to segment into one utterance.
    #[test]
    fn refuses_a_constituency_parse_with_no_tree_or_an_unreadable_tree() {
        for trees in [vec![], vec!["not a tree".to_string()]] {
            let error = admit_worker_item(
                &two_word_request(),
                &UtsegItemResultV2::Constituency { trees },
            )
            .expect_err("no segmentation without a readable parse");
            assert!(error.contains("utseg constituency parse"), "{error}");
        }
    }

    #[test]
    fn refuses_assignments_that_disagree_with_applied_boundary_evidence() {
        let mut result = boundary_result(2);
        *boundary_parts(&mut result).0 = vec![0, 0];

        let error = admit_worker_item(&two_word_request(), &result)
            .expect_err("evidence and assignments must describe one decision");

        assert!(error.contains("assignments disagree"));
    }

    #[test]
    fn refuses_applied_actions_that_disagree_with_declared_policy() {
        let mut result = boundary_result(2);
        let evidence = boundary_parts(&mut result).1;
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
        let evidence = boundary_parts(&mut result).1;
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
        *boundary_parts(&mut result).0 = vec![0, 0];
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

    /// The check must accept exactly what the local policy itself produces, or
    /// the replay could not read back evidence this build wrote.
    #[test]
    fn a_receipt_the_local_policy_produced_is_accepted() {
        let mut result = boundary_result(2);
        let evidence = boundary_parts(&mut result).1;
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
        *boundary_parts(&mut result).0 = vec![0, 0];
        let request = two_word_request();
        let reapplied = admit_worker_item(&request, &result)
            .expect("baseline evidence")
            .apply_decision_policy(
                &request,
                &LanguageCode3::eng(),
                UtsegDecisionPolicy::ReapplyBoundaryModel(
                    UtsegAdjacencyPolicyRevisionV2::SuppressEarlierAdjacentBoundariesV1,
                ),
            )
            .expect("candidate replay");
        let AdmittedUtsegPrediction::BoundaryModelLocallyReapplied {
            response,
            evidence,
            receipt,
        } = reapplied
        else {
            panic!("the candidate policy produces a locally reapplied prediction")
        };

        // Exactly what the evidence reader reconstructs from a retained
        // artifact: the admitted parts, then the receipt reattached.
        let reconstructed =
            AdmittedUtsegPrediction::BoundaryModelWorkerDeclared { response, evidence }
                .with_local_decision(receipt)
                .expect("a receipt this policy produced must be accepted");
        assert!(matches!(
            reconstructed,
            AdmittedUtsegPrediction::BoundaryModelLocallyReapplied { .. }
        ));
    }

    /// RED FIRST (review item 4): a receipt is a claim, not a label, and it can
    /// arrive from an artifact where nothing checked it. One that does not
    /// explain the evidence it travels with must never become a prediction.
    #[test]
    fn a_receipt_that_does_not_explain_its_evidence_is_refused() {
        let admitted =
            admit_worker_item(&two_word_request(), &boundary_result(2)).expect("baseline evidence");
        let error = admitted
            .with_local_decision(LocalUtsegDecisionReceipt {
                revision: LocalUtsegDecisionPolicyRevision::AdjacencyOnlyV1,
                worker_adjacency_policy_revision:
                    UtsegAdjacencyPolicyRevisionV2::SuppressEarlierAdjacentNonordinaryV1,
                // The fixture's evidence declares the nonordinary policy, so a
                // receipt naming a different local policy explains nothing.
                local_adjacency_policy_revision:
                    UtsegAdjacencyPolicyRevisionV2::SuppressEarlierAdjacentBoundariesV1,
                worker_assignments: vec![0, 1],
                suppressed_split_before_word_indices: Vec::new(),
            })
            .expect_err("an unexplained receipt must be refused");

        assert!(error.contains("adjacency policy"), "{error}");
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
        let result = UtsegItemResultV2::BoundaryModel {
            assignments: vec![0; words.len()],
            boundary_model_evidence: UtsegBoundaryModelEvidenceV2 {
                model_id: "model".into(),
                model_revision: test_commit(),
                normalization_revision: UtsegNormalizationRevisionV2::LowerStripAsciiPunctuationV1,
                adjacency_policy_revision:
                    UtsegAdjacencyPolicyRevisionV2::SuppressEarlierAdjacentNonordinaryV1,
                word_evidence: evidence_words,
            },
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
    #[test]
    fn invalidated_analysis_is_recorded_in_typed_output_without_losing_free_text() {
        let source = include_str!("../../../test-fixtures/eng_hello_world_with_mor_gra_act.cha");
        let mut chat = batchalign_transform::parse_and_validate(
            source,
            talkbank_model::ParseValidateOptions::default().with_validation(),
        )
        .expect("valid existing analyzed CHAT");
        apply_utseg_document(&mut chat, &HashMap::from([(0, vec![0, 1])]))
            .expect("supported partition");
        let text = batchalign_transform::serialize::to_chat_string(&chat);
        assert!(text.contains("input utterance 1 %mor"));
        assert!(text.contains("input utterance 1 %gra"));
        assert!(text.contains("%act:\tsmiles"));
        assert!(!text.contains("%mor:"));
        assert!(!text.contains("%gra:"));
        crate::pipeline::post_validate::PostValidated::gate_owned(
            chat,
            crate::api::ReleasedCommand::Utseg,
        )
        .expect("checked construction of recorded partition");
    }
}
