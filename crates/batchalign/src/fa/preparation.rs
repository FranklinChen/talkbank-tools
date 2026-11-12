//! Consuming source-group preparation before dispatch/cache geometry is frozen.

use std::collections::{HashMap, HashSet};

use crate::api::{LanguageCode3, WorkerLanguage};
use crate::chat_ops::fa::coordinates::FaWindow;
use crate::chat_ops::fa::resource_partition::{
    self, ResourcePlanError, ResourceProbe, ResourceProbeOutcome,
};
use crate::chat_ops::fa::{FaGroup, FaInferItem, FaWord, GroupSpan, GroupUnit, cache_key};
use crate::error::ServerError;
use crate::params::{AudioContext, CachePolicy, FaParams};
use crate::types::engines::FaEngineName;
use crate::types::worker_v2::{
    ExecuteOutcomeRef, ExecuteResponseV2, FaPreparationOutcomeV2, FaResourceConstraintV2,
    TaskResultV2,
};
use crate::worker::artifacts_v2::PreparedArtifactRuntimeV2;
use crate::worker::error::WorkerError;
use crate::worker::fa_result_v2::{FaEngineConstraint, FaFailureClass, FaWorkerFailure};
use crate::worker::pool::{BoundFaPreparation, FaProbeCompletion, GpuTask};
use crate::worker::request_builder_v2::{
    ForcedAlignmentBuildInputV2, ForcedAlignmentRequestBuildErrorV2, PreparedFaCandidateV2,
    PreparedFaRequestIdsV2, build_forced_alignment_request_v2,
};

use super::{FaServices, units::DispatchUnit};
use crate::runner::util::{FileStage, ProgressSender, ProgressUpdate};
mod registry;
use registry::{
    PreparationRegistry, PreparationRegistryError, PreparedInputRef, SelectedPreparations,
};
pub(super) use registry::{PreparationWorkPermit, ResidualPreparations, RetainedPreparation};

/// The immutable run facts are carried by preparation into cache and dispatch.
#[derive(Clone, Copy)]
pub(super) struct FaPreparationContext<'a> {
    pub(super) audio: &'a AudioContext<'a>,
    pub(super) worker_lang: &'a LanguageCode3,
    pub(super) services: FaServices<'a>,
    pub(super) params: &'a FaParams,
}

/// Only source preparation constructs this phase. Geometry and retained
/// execution inputs move together; callers cannot independently replace them.
pub(super) struct PreparedFaPopulation<'a> {
    pub(super) context: FaPreparationContext<'a>,
    pub(super) groups: Vec<FaGroup>,
    pub(super) requests: PreparedFaRequests,
    pub(super) cached: super::units::CandidateCacheReceipts,
    pub(super) retry: SourceRetryPreparation,
}

pub(super) type SourceRetryPreparation =
    ResidualPreparations<crate::chat_ops::fa::SuppliedRetrySource, PreparedFaInput>;

/// Unlike a cache key, a preparation scope also includes stable word identity.
/// Identical inputs share one preparation and execution even when their
/// projections own different output ranges.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) struct RequestScope {
    semantic_key: String,
    word_ids: Vec<String>,
}

impl RequestScope {
    pub(super) fn of(
        words: &[FaWord],
        window: FaWindow,
        context: FaPreparationContext<'_>,
    ) -> Self {
        let texts: Vec<_> = words.iter().map(|word| word.text.clone()).collect();
        Self {
            semantic_key: cache_key(
                &texts,
                context.audio.audio_identity,
                window.audio_start().get(),
                window.end().get(),
                context.params.whisper_text,
                context.params.engine,
            )
            .as_str()
            .to_owned(),
            word_ids: words.iter().map(FaWord::stable_id).collect(),
        }
    }
    pub(super) fn for_unit(unit: &DispatchUnit<'_>) -> Self {
        Self {
            semantic_key: unit.cache_key().as_str().to_owned(),
            word_ids: unit.words().iter().map(FaWord::stable_id).collect(),
        }
    }
}

pub(super) struct PreparedFaRequests {
    entries: HashMap<RequestScope, PreparedFaRequest>,
}

impl PreparedFaRequests {
    /// Consume both inference authority and prepared owners together. Only a
    /// new input receives an execution owner; aliases receive its projection.
    pub(super) fn bind<'a>(
        mut self,
        authorization: super::transport::FaInferenceAuthorization<'a>,
    ) -> Result<Vec<PreparedFaDispatch<'a>>, ServerError> {
        let mut indices: HashMap<RequestScope, usize> = HashMap::new();
        let mut dispatches: Vec<PreparedFaDispatch<'a>> = Vec::new();
        for reply in authorization.into_replies() {
            let scope = RequestScope::for_unit(reply.unit());
            match indices.entry(scope.clone()) {
                std::collections::hash_map::Entry::Occupied(index) => {
                    dispatches[*index.get()].aliases.push(reply)
                }
                std::collections::hash_map::Entry::Vacant(index) => {
                    let prepared = self.entries.remove(&scope).ok_or_else(|| {
                        ServerError::Worker(WorkerError::Protocol(format!(
                            "FA dispatch request {} has no retained preparation",
                            reply.unit().ordinal()
                        )))
                    })?;
                    index.insert(dispatches.len());
                    dispatches.push(PreparedFaDispatch {
                        reply,
                        prepared,
                        aliases: Vec::new(),
                    });
                }
            }
        }
        Ok(dispatches)
    }
}

pub(super) struct PreparedFaDispatch<'a> {
    pub(super) reply: super::units::ReplyTicket<'a>,
    pub(super) prepared: PreparedFaRequest,
    pub(super) aliases: Vec<super::units::ReplyTicket<'a>>,
}

struct MaterializedFaInputs {
    candidate: PreparedFaCandidateV2,
    artifacts: PreparedArtifactRuntimeV2,
}

#[derive(Debug, Clone, Copy)]
enum PreparedFaPath {
    Direct,
    Fallback {
        engine: FaEngineName,
        constraint: FaEngineConstraint,
    },
}

pub(super) enum PreparedFaExecution {
    Direct {
        response: ExecuteResponseV2,
        fallback: Box<FaFallbackInputs>,
    },
    Fallback {
        response: ExecuteResponseV2,
        engine: FaEngineName,
        constraint: FaEngineConstraint,
    },
}

/// Only a direct execution can issue this consuming fallback capability.
/// Its retained descriptors contain no original media path or decoder access.
pub(super) struct FaFallbackInputs {
    inputs: MaterializedFaInputs,
}

pub(super) enum FaFallbackExecution {
    Response(ExecuteResponseV2),
    Limited(FaResourceConstraintV2),
}

impl FaFallbackInputs {
    pub(super) async fn execute(
        self,
        services: FaServices<'_>,
        language: WorkerLanguage,
        engine: FaEngineName,
        semantic_key: &str,
    ) -> Result<FaFallbackExecution, ServerError> {
        let capture = services
            .attention_capture_dir
            .filter(|_| engine == FaEngineName::Whisper)
            .map(|directory| super::attention_capture::request(directory, semantic_key))
            .transpose()?;
        let candidate = self
            .inputs
            .candidate
            .retarget(
                engine,
                format!("fa-fallback-run-{}", uuid::Uuid::new_v4()).into(),
            )
            .with_attention_capture(capture)
            .map_err(|error| ServerError::Worker(WorkerError::Protocol(error.to_string())))?;
        let inputs = MaterializedFaInputs {
            candidate,
            artifacts: self.inputs.artifacts,
        };
        match prepare_backend(services, &inputs, GpuTask::FaFallback, language.clone()).await? {
            BackendPreparation::Prepared(bound) => {
                let response =
                    dispatch_bound(services, &inputs, bound, language, GpuTask::FaFallback).await?;
                Ok(FaFallbackExecution::Response(response))
            }
            BackendPreparation::Refused(response) => Ok(FaFallbackExecution::Response(response)),
            BackendPreparation::Limited(constraint) => Ok(FaFallbackExecution::Limited(constraint)),
        }
    }
}

/// Own the exact PCM/text artifacts until execution and capture admission end.
pub(super) struct PreparedFaRequest {
    inputs: MaterializedFaInputs,
    bound: BoundFaPreparation,
    path: PreparedFaPath,
}

impl PreparedFaRequest {
    pub(super) async fn execute(
        self,
        services: FaServices<'_>,
        language: WorkerLanguage,
    ) -> Result<PreparedFaExecution, ServerError> {
        let task = match self.path {
            PreparedFaPath::Direct => GpuTask::Fa,
            PreparedFaPath::Fallback { .. } => GpuTask::FaFallback,
        };
        let response = dispatch_bound(services, &self.inputs, self.bound, language, task).await?;
        Ok(match self.path {
            PreparedFaPath::Direct => PreparedFaExecution::Direct {
                response,
                fallback: Box::new(FaFallbackInputs {
                    inputs: self.inputs,
                }),
            },
            PreparedFaPath::Fallback { engine, constraint } => PreparedFaExecution::Fallback {
                response,
                engine,
                constraint,
            },
        })
    }
}

async fn dispatch_bound(
    services: FaServices<'_>,
    _inputs: &MaterializedFaInputs,
    bound: BoundFaPreparation,
    language: WorkerLanguage,
    task: GpuTask,
) -> Result<ExecuteResponseV2, ServerError> {
    let route = services
        .pipeline
        .pool
        .gpu_route(services.gpu, task, language)?;
    let capture = bound.plan().binding.attention_capture.clone();
    let id = bound.plan().binding.execution_request_id.clone();
    let response = services
        .pipeline
        .pool
        .dispatch_ready_routed(&route, bound.into_ready())
        .await?;
    if let Some(capture) = capture {
        super::attention_capture::admit_response(&capture, &id, &response).await?;
    }
    Ok(response)
}

struct LiveResourceProbe<'a> {
    context: FaPreparationContext<'a>,
    registry: PreparationRegistry<PreparedFaInput>,
    cached: super::units::CandidateCacheReceipts,
}

pub(super) enum PreparedFaInput {
    Cached(super::units::UnitAnswer),
    Inference(Box<PreparedFaRequest>),
}

enum BackendPreparation {
    Prepared(BoundFaPreparation),
    Limited(FaResourceConstraintV2),
    Refused(ExecuteResponseV2),
}

fn prepare_backend<'run>(
    services: FaServices<'run>,
    inputs: &'run MaterializedFaInputs,
    task: GpuTask,
    language: WorkerLanguage,
) -> futures::future::BoxFuture<'run, Result<BackendPreparation, ServerError>> {
    Box::pin(async move {
        let packet = inputs.candidate.request();
        let route = services
            .pipeline
            .pool
            .gpu_route(services.gpu, task, language)?;
        Ok(
            match services
                .pipeline
                .pool
                .prepare_fa_routed(&route, &packet)
                .await?
            {
                FaProbeCompletion::ModelFree(bound) => BackendPreparation::Prepared(bound),
                FaProbeCompletion::Feasible(feasible) => {
                    BackendPreparation::Prepared(feasible.into_bound())
                }
                FaProbeCompletion::Limited(limited) => {
                    BackendPreparation::Limited(limited.into_constraint())
                }
                FaProbeCompletion::Refused(response) => BackendPreparation::Refused(response),
            },
        )
    })
}

/// The model-preparation phase deliberately has no original media path.
#[derive(Clone, Copy)]
struct ModelPreparationContext<'a> {
    services: FaServices<'a>,
    worker_lang: &'a LanguageCode3,
    params: &'a FaParams,
}

impl ModelPreparationContext<'_> {
    fn capture(
        &self,
        engine: FaEngineName,
        semantic_key: &str,
    ) -> Result<Option<crate::types::worker_v2::WhisperAttentionCaptureRequestV2>, ServerError>
    {
        self.services
            .attention_capture_dir
            .filter(|_| engine == FaEngineName::Whisper)
            .map(|directory| super::attention_capture::request(directory, semantic_key))
            .transpose()
    }

    async fn backend(
        &self,
        inputs: &MaterializedFaInputs,
        task: GpuTask,
    ) -> Result<BackendPreparation, ServerError> {
        prepare_backend(
            self.services,
            inputs,
            task,
            WorkerLanguage::from(self.worker_lang.clone()),
        )
        .await
    }

    async fn prepare(
        self,
        mut inputs: MaterializedFaInputs,
        semantic_key: &str,
    ) -> Result<ResourceProbeOutcome<PreparedFaRequest>, ServerError> {
        let primary = self.backend(&inputs, GpuTask::Fa).await?;
        let (result, path) = match primary {
            BackendPreparation::Refused(ref response) => {
                let cause = preparation_refusal(response)?;
                match (
                    FaWorkerFailure::from_preparation(cause.clone()).class(),
                    self.params.engine.fallback_policy().retry_target(),
                ) {
                    (FaFailureClass::EngineConstraint(constraint), Some(engine)) => {
                        let candidate = inputs
                            .candidate
                            .retarget(
                                engine,
                                format!("fa-fallback-plan-{}", uuid::Uuid::new_v4()).into(),
                            )
                            .with_attention_capture(self.capture(engine, semantic_key)?)
                            .map_err(|error| {
                                ServerError::Worker(WorkerError::Protocol(error.to_string()))
                            })?;
                        inputs = MaterializedFaInputs {
                            candidate,
                            artifacts: inputs.artifacts,
                        };
                        (
                            self.backend(&inputs, GpuTask::FaFallback).await?,
                            PreparedFaPath::Fallback { engine, constraint },
                        )
                    }
                    (FaFailureClass::FileFailure | FaFailureClass::ModelUnavailable, _) => {
                        return Err(FaWorkerFailure::from_preparation(cause)
                            .into_server_error(format_args!("FA request preparation")));
                    }
                    (FaFailureClass::EngineConstraint(_), None) => {
                        (primary, PreparedFaPath::Direct)
                    }
                }
            }
            BackendPreparation::Prepared(_) | BackendPreparation::Limited(_) => {
                (primary, PreparedFaPath::Direct)
            }
        };
        match result {
            BackendPreparation::Prepared(bound) => {
                if matches!(path, PreparedFaPath::Direct)
                    && bound.plan().binding.engine != *self.services.cache_namespace.name()
                {
                    return Err(ServerError::Worker(WorkerError::Protocol(
                        "FA preparation engine differs from its admitted cache namespace".into(),
                    )));
                }
                Ok(ResourceProbeOutcome::Prepared(PreparedFaRequest {
                    inputs,
                    bound,
                    path,
                }))
            }
            BackendPreparation::Limited(constraint) => {
                Ok(ResourceProbeOutcome::Limited(constraint))
            }
            BackendPreparation::Refused(response) => {
                let cause = preparation_refusal(&response)?;
                if matches!(
                    FaWorkerFailure::from_preparation(cause.clone()).class(),
                    FaFailureClass::FileFailure
                ) {
                    return Err(FaWorkerFailure::from_preparation(cause)
                        .into_server_error(format_args!("FA fallback preparation")));
                }
                Ok(ResourceProbeOutcome::Limited(
                    FaResourceConstraintV2::Model { cause },
                ))
            }
        }
    }
}

pub(super) async fn materialize_once(
    request: GroupUnit<'_>,
    context: FaPreparationContext<'_>,
    permit: PreparationWorkPermit,
    semantic_key: &str,
) -> Result<ResourceProbeOutcome<PreparedFaRequest>, ServerError> {
    let artifacts = PreparedArtifactRuntimeV2::new("fa_prepared").map_err(|error| {
        ServerError::Validation(format!("cannot own FA preparation artifacts: {error}"))
    })?;
    let id = format!("fa-plan-{}", permit.attempt());
    let ids = PreparedFaRequestIdsV2::new(id.clone(), format!("{id}-text"), format!("{id}-audio"));
    let words = request.words();
    let item = FaInferItem {
        words: words.iter().map(|word| word.text.clone()).collect(),
        word_ids: words.iter().map(FaWord::stable_id).collect(),
        word_utterance_indices: words
            .iter()
            .map(|word| word.utterance_index.raw())
            .collect(),
        word_utterance_word_indices: words
            .iter()
            .map(|word| word.utterance_word_index.raw())
            .collect(),
        audio_path: context.audio.audio_path.to_string_lossy().into_owned(),
        window: request.window(),
        whisper_text: context.params.whisper_text,
    };
    let candidate = match build_forced_alignment_request_v2(
        artifacts.store(),
        ForcedAlignmentBuildInputV2 {
            ids: &ids,
            infer_item: &item,
            engine: context.params.engine,
        },
    )
    .await
    {
        Ok(candidate) => candidate,
        Err(ForcedAlignmentRequestBuildErrorV2::EmptyAudioSegment(segment)) => {
            return Ok(ResourceProbeOutcome::Limited(
                FaResourceConstraintV2::EmptyDecodedAudio {
                    shortfall: segment.reason.into_wire(),
                },
            ));
        }
        Err(other) => {
            return Err(ServerError::Validation(format!(
                "cannot prepare source-bound FA request: {other}"
            )));
        }
    };
    let model = ModelPreparationContext {
        services: context.services,
        worker_lang: context.worker_lang,
        params: context.params,
    };
    let candidate = candidate
        .with_attention_capture(model.capture(context.params.engine, semantic_key)?)
        .map_err(|error| ServerError::Worker(WorkerError::Protocol(error.to_string())))?;
    model
        .prepare(
            MaterializedFaInputs {
                candidate,
                artifacts,
            },
            semantic_key,
        )
        .await
}

impl LiveResourceProbe<'_> {
    async fn prepare_window(
        &mut self,
        request: GroupUnit<'_>,
    ) -> Result<ResourceProbeOutcome<PreparedInputRef>, ServerError> {
        let scope = RequestScope::of(request.words(), request.window(), self.context);
        let semantic_key = scope.semantic_key.clone();
        let context = self.context;
        let availability = self.cached.lookup(request, context).await;
        self.registry.prepare(scope, move |permit| async move {
            match availability {
                super::units::CandidateCacheAvailability::Admitted(answer) => Ok(ResourceProbeOutcome::Prepared(PreparedFaInput::Cached(answer))),
                super::units::CandidateCacheAvailability::Miss => Ok(match materialize_once(request, context, permit, &semantic_key).await? {
                    ResourceProbeOutcome::Prepared(prepared) => ResourceProbeOutcome::Prepared(PreparedFaInput::Inference(Box::new(prepared))),
                    ResourceProbeOutcome::Limited(constraint) => ResourceProbeOutcome::Limited(constraint),
                }),
            }
        }).await.map_err(|error| match error {
            PreparationRegistryError::Operation(error) => error,
            PreparationRegistryError::UnsettledClaim { attempt } => ServerError::Worker(WorkerError::Protocol(format!("FA preparation {attempt} was already claimed without a settled report; abandon this epoch instead of repeating work"))),
        })
    }
}

fn preparation_refusal(
    response: &ExecuteResponseV2,
) -> Result<crate::types::worker_v2::FaPreparationRefusalV2, ServerError> {
    match response.read() {
        ExecuteOutcomeRef::Success(TaskResultV2::FaPreparation(result)) => match &result.outcome {
            FaPreparationOutcomeV2::Refused { cause } => Ok(cause.clone()),
            FaPreparationOutcomeV2::Prepared { .. } => Err(ServerError::Worker(
                WorkerError::Protocol("expected a preparation refusal, received a plan".into()),
            )),
        },
        ExecuteOutcomeRef::Success(_) | ExecuteOutcomeRef::Failed { .. } => {
            Err(ServerError::Worker(WorkerError::Protocol(
                "preparation refusal lost its original typed result".into(),
            )))
        }
    }
}

impl ResourceProbe for LiveResourceProbe<'_> {
    type Prepared = PreparedInputRef;
    type Error = ServerError;
    async fn prepare(
        &mut self,
        request: GroupUnit<'_>,
    ) -> Result<ResourceProbeOutcome<Self::Prepared>, ServerError> {
        self.prepare_window(request).await
    }
}

pub(super) fn prepare_population<'a: 'run, 'run>(
    groups: Vec<FaGroup>,
    sources: crate::chat_ops::fa::SuppliedRetrySources,
    chat: &'run crate::chat_ops::ChatFile,
    reusable: &'run HashSet<usize>,
    anchors: &'run crate::chat_ops::fa::AnchorIndex,
    context: FaPreparationContext<'a>,
    progress: Option<&'run ProgressSender>,
) -> futures::future::BoxFuture<'run, Result<PreparedFaPopulation<'a>, ServerError>> {
    Box::pin(async move {
        let mut planned = resource_partition::FilePlan::default();
        let retain = sources
            .into_sources()
            .into_iter()
            .map(|source| {
                (
                    RequestScope::of(source.words(), source.window(), context),
                    source,
                )
            })
            .collect();
        if let Some(tx) = progress {
            let _ = tx.send(ProgressUpdate::new(FileStage::CheckingCache, None, None));
        }
        let cached = super::units::collect_candidate_cache(&groups, context).await;
        if context.params.cache_policy == CachePolicy::RequireCache {
            return Ok(PreparedFaPopulation {
                context,
                groups,
                requests: PreparedFaRequests {
                    entries: HashMap::new(),
                },
                cached,
                retry: PreparationRegistry::new()
                    .freeze_retaining(SelectedPreparations::new(), retain)
                    .map_err(|_| unknown_selection())?
                    .1,
            });
        }
        let mut probe = LiveResourceProbe {
            context,
            registry: PreparationRegistry::new(),
            cached,
        };
        let mut selected = SelectedPreparations::new();
        let total = groups.len() as i64;
        for (index, input) in resource_partition::planning_queue(groups).enumerate() {
            if let Some(tx) = progress {
                let _ = tx.send(ProgressUpdate::new(
                    FileStage::PreparingAlignment,
                    Some(index as i64),
                    Some(total),
                ));
            }
            let group = input.group();
            if group.is_reusable_from(reusable)
                && super::incremental::collect_preserved_group_timings(chat, group).is_some()
            {
                planned.keep(input.into_group());
                continue;
            }
            if let GroupSpan::Single(single) = group.span() {
                let whole = single.request();
                if probe
                    .cached
                    .contains(whole.words(), whole.window(), context)
                {
                    planned.keep(input.into_group());
                    continue;
                }
            }
            planned.plan(
                resource_partition::prepare_group(
                    input,
                    anchors,
                    &mut probe,
                    |_request, reference| selected.select(reference),
                )
                .await
                .map_err(plan_error)?,
            );
        }
        let planned = planned.finish().map_err(host_too_small)?;
        let (entries, residual) = probe
            .registry
            .freeze_retaining(selected, retain)
            .map_err(|_| unknown_selection())?;
        let entries = entries
            .into_iter()
            .filter_map(|(scope, input)| match input {
                PreparedFaInput::Cached(_) => None,
                PreparedFaInput::Inference(prepared) => Some((scope, *prepared)),
            })
            .collect();
        Ok(PreparedFaPopulation {
            context,
            groups: planned,
            requests: PreparedFaRequests { entries },
            cached: probe.cached,
            retry: residual,
        })
    })
}

fn unknown_selection() -> ServerError {
    ServerError::Worker(WorkerError::Protocol(
        "FA geometry selected preparation from another epoch".into(),
    ))
}

fn plan_error(error: ResourcePlanError<ServerError>) -> ServerError {
    match error {
        ResourcePlanError::Probe(error) => error,
        ResourcePlanError::InvalidCut(_) => ServerError::Worker(WorkerError::Protocol(
            "FA planner attempted a cut not supported by its source population".into(),
        )),
    }
}

/// The host admitted nothing of the file: the same final refusal a worker
/// reservation gets, actionable and never retried, since the same host
/// cannot offer more.
fn host_too_small(host: resource_partition::HostTooSmall) -> ServerError {
    ServerError::Worker(WorkerError::MemoryGuard(
        crate::worker::memory_guard::MemoryGuardError::ExceedsHost(crate::host_memory::NeverFits {
            label: format!("FA model {}", host.engine),
            needed_mb: host.required_mb,
            ceiling_mb: host.ceiling_mb,
            reserve_mb: host.reserve_mb,
        }),
    ))
}

#[cfg(test)]
mod tests;
