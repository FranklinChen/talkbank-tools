//! Input-bound preparation, before the only transition that grants execution.

use super::fa_capacity::{FaCapacityRefusal, FaHostAssessment, HostFeasibleFaDemand};
use crate::types::worker_v2::{
    ArtifactRefV2, ExecuteOutcomeRef, ExecuteRequestV2, ExecuteResponseV2, FaPreparationOutcomeV2,
    FaPreparationPlanV2, FaRequestMemoryV2, FaRequestPhaseV2, ForcedAlignmentRequestV2,
    InferenceTaskV2, TaskRequestV2, TaskResultV2, WorkerRequestIdV2,
};
use crate::worker::error::WorkerError;

/// A dispatch either needs no FA preparation or owns its pending exchange.
#[derive(Debug)]
pub enum DispatchPreparation {
    /// A non-FA request, metadata query, or already bound FA execution.
    Ready(ReadyRequest),
    /// Immutable FA inputs whose loaded-model preparation is still pending.
    ForcedAlignment(FaPreparationExchange),
}

/// A refusal carries the worker's original typed result and measured elapsed
/// time. It grants no execution capability and is never rewritten by the host.
#[derive(Debug)]
pub enum FaPreparationCompletion {
    /// Candidate and loaded-model report whose binding was admitted.
    Ready(BoundFaPreparation),
    /// The worker's original refusal, without execution capability.
    Refused(ExecuteResponseV2),
}

/// A successful report bound to its immutable candidate. Keeping this narrow
/// state separate lets planning inspect demand without granting execution.
#[derive(Debug)]
pub struct BoundFaPreparation {
    candidate: Box<FaCandidate>,
    plan: Box<FaPreparationPlanV2>,
}

/// Preparation plus the same physical feasibility policy execution uses.
/// Every outcome still lacks an execution lease.
#[derive(Debug)]
pub(crate) enum FaProbeCompletion {
    ModelFree(BoundFaPreparation),
    Feasible(PhysicallyFeasibleFaPreparation),
    Limited(PhysicallyLimitedFaPreparation),
    Refused(ExecuteResponseV2),
}

#[derive(Debug)]
pub(crate) struct PhysicallyFeasibleFaPreparation {
    preparation: BoundFaPreparation,
    _feasibility: HostFeasibleFaDemand,
}

impl PhysicallyFeasibleFaPreparation {
    pub(crate) fn into_bound(self) -> BoundFaPreparation {
        self.preparation
    }
}

#[derive(Debug)]
pub(crate) struct PhysicallyLimitedFaPreparation {
    preparation: BoundFaPreparation,
    refusal: FaCapacityRefusal,
}

impl PhysicallyLimitedFaPreparation {
    pub(crate) fn into_constraint(self) -> crate::types::worker_v2::FaResourceConstraintV2 {
        crate::types::worker_v2::FaResourceConstraintV2::HostMemory {
            engine: self.preparation.plan.binding.engine,
            required_mb: self.refusal.required.0,
            reserve_mb: self.refusal.reserve.0,
            total_mb: self.refusal.total.0,
        }
    }
}

impl BoundFaPreparation {
    pub(crate) fn plan(&self) -> &FaPreparationPlanV2 {
        &self.plan
    }

    /// Advance only when ready to request an execution lease. Planning never
    /// substitutes its capacity observation for that lease-owning transition.
    pub fn into_ready(self) -> ReadyRequest {
        let memory = self.plan.memory.clone();
        let request = self.candidate.wire(
            self.candidate.request_id.clone(),
            FaRequestPhaseV2::Execute { plan: self.plan },
        );
        ReadyRequest {
            request,
            memory: RequestMemory::ForcedAlignment(memory),
        }
    }
}

/// The request shape the lease-granting transition accepts. Constructors are
/// private: an arbitrary execute packet is not a prepared request.
#[derive(Debug)]
pub struct ReadyRequest {
    request: ExecuteRequestV2,
    memory: RequestMemory,
}

#[derive(Debug)]
enum RequestMemory {
    Ordinary,
    MetadataOnly,
    ForcedAlignment(FaRequestMemoryV2),
}

impl ReadyRequest {
    pub(super) fn request(&self) -> &ExecuteRequestV2 {
        &self.request
    }

    /// Only a bound preparation reply can supply FA memory evidence.
    pub(super) fn fa_memory(&self) -> Option<&FaRequestMemoryV2> {
        match &self.memory {
            RequestMemory::ForcedAlignment(memory) => Some(memory),
            RequestMemory::Ordinary | RequestMemory::MetadataOnly => None,
        }
    }
}

impl DispatchPreparation {
    /// Inspect routing/loading identity without exposing an execution constructor.
    /// The preparation query names the same task, backend and artifacts as its
    /// candidate; only the phase and RPC id differ.
    pub(super) fn request(&self) -> &ExecuteRequestV2 {
        match self {
            Self::Ready(ready) => ready.request(),
            Self::ForcedAlignment(exchange) => &exchange.query,
        }
    }

    /// Preserve malformed task/payload pairs for native validation. An FA
    /// preparation exchange may reshape phases only of an actual FA request.
    pub fn begin(request: &ExecuteRequestV2) -> Result<Self, WorkerError> {
        match &request.payload {
            TaskRequestV2::ForcedAlignment(payload)
                if request.task == InferenceTaskV2::ForcedAlignment =>
            {
                match payload.phase {
                    FaRequestPhaseV2::Candidate => {
                        let candidate = FaCandidate {
                            request_id: request.request_id.clone(),
                            payload: payload.clone(),
                            attachments: request.attachments.clone(),
                        };
                        let query = candidate.wire(
                            format!("{}:prepare", request.request_id).into(),
                            FaRequestPhaseV2::Prepare {
                                execution_request_id: request.request_id.clone(),
                            },
                        );
                        Ok(Self::ForcedAlignment(FaPreparationExchange {
                            candidate,
                            query,
                        }))
                    }
                    FaRequestPhaseV2::Prepare { .. } => Ok(Self::Ready(ReadyRequest {
                        request: request.clone(),
                        memory: RequestMemory::MetadataOnly,
                    })),
                    FaRequestPhaseV2::Execute { .. } => Err(WorkerError::Protocol(
                        "FA execution requires a bound candidate, not a raw execution packet"
                            .into(),
                    )),
                }
            }
            TaskRequestV2::ForcedAlignment(_)
            | TaskRequestV2::Morphosyntax(_)
            | TaskRequestV2::Utseg(_)
            | TaskRequestV2::Translate(_)
            | TaskRequestV2::Coref(_)
            | TaskRequestV2::WordSegmentation(_)
            | TaskRequestV2::Asr(_)
            | TaskRequestV2::Speaker(_)
            | TaskRequestV2::SpeakerEmbedding(_)
            | TaskRequestV2::Opensmile(_)
            | TaskRequestV2::Avqi(_) => Ok(Self::Ready(ReadyRequest {
                request: request.clone(),
                memory: RequestMemory::Ordinary,
            })),
        }
    }
}

/// The immutable candidate and the query whose reply can advance it.
#[derive(Debug)]
pub struct FaPreparationExchange {
    candidate: FaCandidate,
    query: ExecuteRequestV2,
}

#[derive(Debug)]
struct FaCandidate {
    request_id: WorkerRequestIdV2,
    payload: ForcedAlignmentRequestV2,
    attachments: Vec<ArtifactRefV2>,
}

impl FaCandidate {
    fn wire(&self, request_id: WorkerRequestIdV2, phase: FaRequestPhaseV2) -> ExecuteRequestV2 {
        let mut payload = self.payload.clone();
        payload.phase = phase;
        ExecuteRequestV2 {
            request_id,
            task: InferenceTaskV2::ForcedAlignment,
            payload: TaskRequestV2::ForcedAlignment(payload),
            attachments: self.attachments.clone(),
        }
    }
}

impl FaPreparationExchange {
    /// A metadata-only request may run under the already-held model lease.
    pub fn query(&self) -> ReadyRequest {
        ReadyRequest {
            request: self.query.clone(),
            memory: RequestMemory::MetadataOnly,
        }
    }

    /// Consume the exchange. The report cannot be applied to another request.
    pub fn bind(self, reply: ExecuteResponseV2) -> Result<FaPreparationCompletion, WorkerError> {
        if reply.request_id() != &self.query.request_id {
            return Err(WorkerError::Protocol(
                "FA preparation reply id mismatch".into(),
            ));
        }
        let plan = match reply.read() {
            ExecuteOutcomeRef::Success(TaskResultV2::FaPreparation(result)) => {
                match &result.outcome {
                    FaPreparationOutcomeV2::Prepared { plan } => plan.clone(),
                    FaPreparationOutcomeV2::Refused { .. } => {
                        return Ok(FaPreparationCompletion::Refused(reply));
                    }
                }
            }
            ExecuteOutcomeRef::Success(_) => {
                return Err(WorkerError::Protocol(
                    "FA preparation returned inference data".into(),
                ));
            }
            ExecuteOutcomeRef::Failed { code, message } => {
                return Err(WorkerError::RequestRefused(format!(
                    "FA preparation {code:?}: {message}"
                )));
            }
        };
        let candidate = self.candidate;
        let payload = &candidate.payload;
        let audio = candidate
            .attachments
            .iter()
            .find_map(|attachment| match attachment {
                ArtifactRefV2::PreparedAudio(audio) if audio.id == payload.audio_ref_id => {
                    Some(audio)
                }
                _ => None,
            })
            .ok_or_else(|| {
                WorkerError::Protocol("FA preparation has no matching audio descriptor".into())
            })?;
        let binding = &plan.binding;
        if binding.execution_request_id != candidate.request_id
            || binding.backend != payload.backend
            || binding.payload_ref_id != payload.payload_ref_id
            || binding.audio_ref_id != payload.audio_ref_id
            || binding.samples.get() != audio.frame_count.0
            || binding.sample_rate_hz != audio.sample_rate_hz
            || binding.text_mode != payload.text_mode
            || binding.attention_capture != payload.attention_capture
        {
            return Err(WorkerError::Protocol(
                "FA preparation does not bind this candidate's inputs".into(),
            ));
        }
        Ok(FaPreparationCompletion::Ready(BoundFaPreparation {
            candidate: Box::new(candidate),
            plan: Box::new(plan),
        }))
    }
}

impl super::WorkerPool {
    /// Obtain input-bound allocation evidence over the existing routed
    /// transport. The report is available to planning, but no inference lease
    /// is acquired and no execution packet is sent.
    pub(crate) async fn prepare_fa_routed(
        &self,
        route: &super::GpuRoute,
        candidate: &ExecuteRequestV2,
    ) -> Result<FaProbeCompletion, WorkerError> {
        let DispatchPreparation::ForcedAlignment(exchange) = DispatchPreparation::begin(candidate)?
        else {
            return Err(WorkerError::Protocol(
                "FA preparation probe requires a candidate".into(),
            ));
        };
        let query = exchange.query();
        let reply = self.dispatch_routed_v2(route, query.request()).await?;
        match exchange.bind(reply)? {
            FaPreparationCompletion::Refused(reply) => Ok(FaProbeCompletion::Refused(reply)),
            FaPreparationCompletion::Ready(preparation) => {
                match self
                    .load_preparation()
                    .admission
                    .assess_fa_memory(&preparation.plan.memory)?
                {
                    FaHostAssessment::ModelFree => Ok(FaProbeCompletion::ModelFree(preparation)),
                    FaHostAssessment::Fits(feasibility) => Ok(FaProbeCompletion::Feasible(
                        PhysicallyFeasibleFaPreparation {
                            preparation,
                            _feasibility: feasibility,
                        },
                    )),
                    FaHostAssessment::Exceeds(refusal) => {
                        Ok(FaProbeCompletion::Limited(PhysicallyLimitedFaPreparation {
                            preparation,
                            refusal,
                        }))
                    }
                }
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
pub(crate) mod tests {
    use super::*;
    use crate::api::{NonNegativeSeconds, ReportedEngineName};
    use crate::types::worker_v2::{
        FaPreparationBindingV2, FaPreparationPlanV2, FaPreparationResultV2, FaSampleCountV2,
    };

    fn candidate() -> ExecuteRequestV2 {
        serde_json::from_str(include_str!("../../../../../tests/fixtures/worker_protocol_v2/execute_request_fa_wave2vec_prepared_audio.json")).expect("candidate fixture")
    }

    #[test]
    fn preparation_cannot_rewrite_a_mismatched_task_into_a_valid_fa_request() {
        let mut request = candidate();
        request.task = InferenceTaskV2::Morphosyntax;
        let DispatchPreparation::Ready(ready) =
            DispatchPreparation::begin(&request).expect("the worker owns task/payload validation")
        else {
            panic!("preparation must not normalize a contradictory task into FA")
        };
        assert_eq!(
            ready.request(),
            &request,
            "preserve the exact malformed packet for native validation"
        );
    }

    #[tokio::test]
    async fn worker_roundtrip_routed_metadata_preparation_returns_a_bound_report_without_inference()
    {
        use crate::api::{LanguageCode3, WorkerLanguage};
        use crate::types::engines::FaEngineName;
        use crate::worker::pool::{GpuModelPlan, GpuTask, PoolConfig, WorkerPool};

        let pool = WorkerPool::new(PoolConfig {
            test_echo: true,
            ..PoolConfig::for_tests()
        });
        let request = candidate();
        let route = pool
            .gpu_route(
                &GpuModelPlan::alignment(FaEngineName::Wave2Vec, None),
                GpuTask::Fa,
                WorkerLanguage::from(LanguageCode3::eng()),
            )
            .expect("declared FA route");
        let outcome = pool.prepare_fa_routed(&route, &request).await;
        let FaProbeCompletion::ModelFree(prepared) = outcome.expect("metadata roundtrip") else {
            panic!("an echo host reports model-free preparation, never inference")
        };
        assert_eq!(
            prepared.plan().binding.execution_request_id,
            request.request_id
        );
        assert!(matches!(prepared.plan().memory, FaRequestMemoryV2::NoModel));
        let response = pool
            .dispatch_ready_routed(&route, prepared.into_ready())
            .await;
        pool.shutdown().await;
        let response = response.expect("bound preparation consumes a fresh execution lease");
        assert_eq!(response.request_id(), &request.request_id);
        assert!(matches!(
            response.read(),
            ExecuteOutcomeRef::Success(TaskResultV2::IndexedWordTimingResult(_))
        ));
    }

    #[test]
    fn an_explicit_preparation_query_has_no_inference_memory_demand() {
        let pending = exchange(&candidate());
        let query = pending.query();
        let DispatchPreparation::Ready(metadata) = DispatchPreparation::begin(query.request())
            .expect("a metadata-only probe can use the shared transport")
        else {
            panic!("metadata must not start another preparation exchange")
        };
        assert!(metadata.fa_memory().is_none());
        assert!(matches!(
            metadata.request().payload,
            TaskRequestV2::ForcedAlignment(ForcedAlignmentRequestV2 {
                phase: FaRequestPhaseV2::Prepare { .. },
                ..
            })
        ));
    }

    pub(crate) fn prepared_request(memory: FaRequestMemoryV2) -> ReadyRequest {
        let pending = exchange(&candidate());
        let prepared = plan(&pending, memory);
        let response = reply(&pending, prepared);
        match pending
            .bind(response)
            .expect("matching declared preparation")
        {
            FaPreparationCompletion::Ready(ready) => ready.into_ready(),
            FaPreparationCompletion::Refused(_) => panic!("declared fixture was refused"),
        }
    }

    pub(super) fn exchange(request: &ExecuteRequestV2) -> FaPreparationExchange {
        match DispatchPreparation::begin(request).expect("candidate") {
            DispatchPreparation::ForcedAlignment(exchange) => exchange,
            DispatchPreparation::Ready(_) => panic!("FA must be prepared"),
        }
    }

    pub(super) fn plan(
        exchange: &FaPreparationExchange,
        memory: FaRequestMemoryV2,
    ) -> FaPreparationPlanV2 {
        let candidate = &exchange.candidate;
        let audio = candidate
            .attachments
            .iter()
            .find_map(|item| match item {
                ArtifactRefV2::PreparedAudio(audio) => Some(audio),
                _ => None,
            })
            .expect("audio");
        FaPreparationPlanV2 {
            binding: FaPreparationBindingV2 {
                execution_request_id: candidate.request_id.clone(),
                backend: candidate.payload.backend,
                engine: serde_json::from_str::<ReportedEngineName>("\"fixture-mms\"")
                    .expect("engine"),
                payload_ref_id: candidate.payload.payload_ref_id.clone(),
                audio_ref_id: candidate.payload.audio_ref_id.clone(),
                samples: FaSampleCountV2::try_from(audio.frame_count.0).expect("samples"),
                sample_rate_hz: audio.sample_rate_hz,
                text_mode: candidate.payload.text_mode,
                attention_capture: candidate.payload.attention_capture.clone(),
            },
            memory,
        }
    }

    pub(super) fn reply(
        exchange: &FaPreparationExchange,
        plan: FaPreparationPlanV2,
    ) -> ExecuteResponseV2 {
        ExecuteResponseV2::success(
            exchange.query.request_id.clone(),
            TaskResultV2::FaPreparation(FaPreparationResultV2 {
                outcome: FaPreparationOutcomeV2::Prepared { plan },
            }),
            NonNegativeSeconds::try_from(0.0).expect("elapsed"),
        )
    }

    #[test]
    fn only_a_bound_preparation_reply_produces_an_execution_packet() {
        let request = candidate();
        let pending = exchange(&request);
        let query = pending.query();
        assert_ne!(query.request().request_id, request.request_id);
        assert!(query.fa_memory().is_none(), "metadata is not inference");
        let plan = plan(&pending, FaRequestMemoryV2::NoModel);
        let response = reply(&pending, plan);
        let FaPreparationCompletion::Ready(ready) = pending.bind(response).expect("bound") else {
            panic!("the matching fixture is prepared")
        };
        let ready = ready.into_ready();
        assert!(matches!(
            ready.fa_memory(),
            Some(FaRequestMemoryV2::NoModel)
        ));
        assert!(
            matches!(
                DispatchPreparation::begin(ready.request()),
                Err(WorkerError::Protocol(_))
            ),
            "a raw execution packet cannot bypass the transition"
        );
    }

    #[test]
    fn a_plan_for_other_samples_cannot_admit_the_candidate() {
        let request = candidate();
        let pending = exchange(&request);
        let mut wrong = plan(&pending, FaRequestMemoryV2::NoModel);
        wrong.binding.samples = FaSampleCountV2::try_from(wrong.binding.samples.get() + 1).unwrap();
        let response = reply(&pending, wrong);
        assert!(matches!(
            pending.bind(response),
            Err(WorkerError::Protocol(_))
        ));
    }

    #[test]
    fn preparation_refusal_preserves_the_typed_cause_and_worker_measurement() {
        use crate::types::worker_v2::{FaConstraintUnitV2, FaPreparationRefusalV2};
        let pending = exchange(&candidate());
        let response = ExecuteResponseV2::success(
            pending.query.request_id.clone(),
            TaskResultV2::FaPreparation(FaPreparationResultV2 {
                outcome: FaPreparationOutcomeV2::Refused {
                    cause: FaPreparationRefusalV2::ModelConstraint {
                        unit: FaConstraintUnitV2::EncodedFrames,
                        provided: 39,
                        maximum: 16,
                        engine: serde_json::from_str("\"fixture-mms\"").expect("engine"),
                    },
                },
            }),
            NonNegativeSeconds::try_from(0.25).expect("measured duration fixture"),
        );
        let expected = response.clone();
        let FaPreparationCompletion::Refused(actual) = pending
            .bind(response)
            .expect("a typed refusal is a completion")
        else {
            panic!("a refusal must grant no execution capability")
        };
        assert_eq!(
            actual, expected,
            "the host must not rewrite a measured worker response"
        );
    }
}
