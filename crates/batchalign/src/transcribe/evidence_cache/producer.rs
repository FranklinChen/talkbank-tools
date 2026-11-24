//! Producer observations from the correlated worker result and durable replay.

use serde::{Deserialize, Serialize};

use crate::types::worker_v2::{
    ExecuteResponseV2, SpeakerBackendV2, SpeakerInferenceEvidenceV2, SpeakerLoadedModelGraphV2,
    SpeakerProviderJobIdV2, WorkerRequestIdV2,
};
use crate::worker::speaker_result_v2::{SpeakerResultParseError, parse_speaker_result_v2};

/// Successful speaker payload paired with the request that produced it.
pub struct SpeakerInferenceReply {
    invocation: SpeakerWorkerInvocation,
    evidence: SpeakerInferenceEvidenceV2,
}

impl SpeakerInferenceReply {
    /// Called after the transport binds the terminal response to its request.
    pub(in crate::transcribe) fn admit(
        response: ExecuteResponseV2,
    ) -> Result<Self, SpeakerResultParseError> {
        let request_id = response.request_id().clone();
        let result = parse_speaker_result_v2(response)?;
        Ok(Self {
            invocation: SpeakerWorkerInvocation {
                request_id,
                server_build: crate::build_hash().to_owned(),
            },
            evidence: result.evidence,
        })
    }

    pub(super) fn into_parts(self) -> (SpeakerWorkerInvocation, SpeakerInferenceEvidenceV2) {
        (self.invocation, self.evidence)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(super) struct SpeakerWorkerInvocation {
    request_id: WorkerRequestIdV2,
    server_build: String,
}

/// Evidence the adapter actually returned. Hosted job identity does not
/// identify its weights. Older local workers did not report a loaded graph.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(super) enum ObservedSpeakerModel {
    HostedJob {
        provider_job_id: SpeakerProviderJobIdV2,
    },
    LocalGraph {
        graph: SpeakerLoadedModelGraphV2,
    },
    LocalGraphUnreported,
}

impl ObservedSpeakerModel {
    fn from_result(result: &SpeakerInferenceEvidenceV2) -> Self {
        match result {
            SpeakerInferenceEvidenceV2::PyannoteAi { job_id, .. } => Self::HostedJob {
                provider_job_id: job_id.clone(),
            },
            SpeakerInferenceEvidenceV2::Pyannote {
                loaded_graph: Some(graph),
                ..
            } => Self::LocalGraph {
                graph: graph.clone(),
            },
            SpeakerInferenceEvidenceV2::Pyannote {
                loaded_graph: None, ..
            } => Self::LocalGraphUnreported,
        }
    }

    const fn backend(&self) -> SpeakerBackendV2 {
        match self {
            Self::HostedJob { .. } => SpeakerBackendV2::PyannoteAi,
            Self::LocalGraph { .. } | Self::LocalGraphUnreported => SpeakerBackendV2::Pyannote,
        }
    }
}

/// Origin of reusable speaker evidence, independent of the current cache
/// decision. Only the worker result or a retained cache envelope supplies it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(super) enum SpeakerEvidenceProducer {
    WorkerReply {
        invocation: SpeakerWorkerInvocation,
        model: ObservedSpeakerModel,
    },
    RawWithoutInvocation {
        model: ObservedSpeakerModel,
    },
    Unrecorded,
}

impl SpeakerEvidenceProducer {
    pub(super) fn from_raw(
        invocation: Option<SpeakerWorkerInvocation>,
        result: &SpeakerInferenceEvidenceV2,
    ) -> Self {
        let model = ObservedSpeakerModel::from_result(result);
        match invocation {
            Some(invocation) => Self::WorkerReply { invocation, model },
            None => Self::RawWithoutInvocation { model },
        }
    }

    pub(super) fn matches_backend(&self, backend: SpeakerBackendV2) -> bool {
        match self {
            Self::WorkerReply { model, .. } | Self::RawWithoutInvocation { model } => {
                model.backend() == backend
            }
            Self::Unrecorded => true,
        }
    }
}
