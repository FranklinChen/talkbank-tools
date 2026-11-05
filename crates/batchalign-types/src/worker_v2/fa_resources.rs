//! Request-bound FA preparation evidence. A report is not an execution permit.

use std::num::NonZeroU64;

use serde::{Deserialize, Serialize};

use crate::api::ReportedEngineName;

use super::{
    FaBackendV2, FaTextModeV2, SampleRateHzV2, WhisperAttentionCaptureRequestV2,
    WorkerArtifactIdV2, WorkerRequestIdV2,
};

macro_rules! positive_quantity {
    ($name:ident, $description:literal) => {
        #[doc = $description]
        #[derive(
            Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema,
        )]
        #[serde(transparent)]
        pub struct $name(NonZeroU64);

        impl $name {
            /// The producer's positive count, in this type's declared unit.
            pub const fn get(self) -> u64 {
                self.0.get()
            }
        }

        impl TryFrom<u64> for $name {
            type Error = FaResourceQuantityError;
            fn try_from(value: u64) -> Result<Self, Self::Error> {
                NonZeroU64::new(value)
                    .map(Self)
                    .ok_or(FaResourceQuantityError {
                        quantity: stringify!($name),
                    })
            }
        }
    };
}

/// A resource population must contain an actual positive quantity.
#[derive(Debug, thiserror::Error)]
#[error("{quantity} must be positive")]
pub struct FaResourceQuantityError {
    quantity: &'static str,
}

positive_quantity!(
    FaWorkspaceBytesV2,
    "Additional request workspace, in bytes."
);
positive_quantity!(FaModelBytesV2, "Persistent model tensor storage, in bytes.");
positive_quantity!(FaSampleCountV2, "Prepared mono audio samples.");

/// The actual request and prepared inputs whose allocation was projected.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FaPreparationBindingV2 {
    /// The execution request being prepared, not the preparation RPC's id.
    pub execution_request_id: WorkerRequestIdV2,
    /// The model family the loaded host admitted.
    pub backend: FaBackendV2,
    /// The loaded algorithm/model identity used by FA caches.
    pub engine: ReportedEngineName,
    /// Immutable prepared source words.
    pub payload_ref_id: WorkerArtifactIdV2,
    /// Immutable prepared audio window.
    pub audio_ref_id: WorkerArtifactIdV2,
    /// Actual prepared samples, not an inferred duration.
    pub samples: FaSampleCountV2,
    /// The prepared waveform's actual sample rate.
    pub sample_rate_hz: SampleRateHzV2,
    /// The exact transcript shaping used to prepare model labels.
    pub text_mode: FaTextModeV2,
    /// Diagnostic capture changes output allocations and cannot be substituted.
    pub attention_capture: Option<WhisperAttentionCaptureRequestV2>,
}

/// How additional allocation demand was obtained. No duration proxy exists.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum FaAllocationBasisV2 {
    /// Virtual tensor lifetimes plus the qualified native CPU CTC layout.
    MmsCpu {
        /// PyTorch that performed the virtual projection.
        torch_version: String,
        /// Source revision of the installed native CTC implementation.
        torchaudio_revision: String,
    },
    /// Loaded Whisper's encoder/decoder configuration and virtual tensors.
    Whisper {
        /// PyTorch that performed the projection.
        torch_version: String,
        /// Transformers that owns the loaded model implementation.
        transformers_version: String,
    },
    /// Loaded Qwen encoder/tokenizer configuration and virtual tensors.
    Qwen {
        /// PyTorch that performed the projection.
        torch_version: String,
        /// Transformers that owns the loaded model implementation.
        transformers_version: String,
        /// Qualified native component graph used for shape projection.
        allocation_graph: String,
    },
}

/// Preparation's memory evidence. Model-free test transports say so explicitly.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum FaRequestMemoryV2 {
    /// No model runs. Admission must independently establish test-echo hosting.
    NoModel,
    /// A real loaded model's request-shaped demand, before host admission.
    Projected {
        /// Additional memory for this request, excluding already leased weights.
        workspace: FaWorkspaceBytesV2,
        /// Model storage needed alongside that workspace.
        model: FaModelBytesV2,
        /// Named derivation and native implementation identity.
        basis: FaAllocationBasisV2,
    },
}

/// The worker's report, still requiring request/model binding and a host lease.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FaPreparationPlanV2 {
    /// Its complete input/model scope.
    pub binding: FaPreparationBindingV2,
    /// Memory demand, never permission to allocate it.
    pub memory: FaRequestMemoryV2,
}

/// A candidate cannot execute; preparation reports demand before admission.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum FaRequestPhaseV2 {
    /// Immutable inputs awaiting the loaded worker's preparation.
    Candidate,
    /// A metadata-only preparation RPC for this later execution request.
    Prepare {
        /// The candidate's id; the preparation RPC has a distinct id.
        execution_request_id: WorkerRequestIdV2,
    },
    /// Prepared inputs, still requiring a lease-owning host execution permit.
    Execute {
        /// The loaded worker's input-bound allocation evidence.
        plan: Box<FaPreparationPlanV2>,
    },
}

/// The quantities an actual model constraint can limit.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Serialize,
    Deserialize,
    schemars::JsonSchema,
    utoipa::ToSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum FaConstraintUnitV2 {
    /// Input waveform samples.
    AudioSamples,
    /// Encoded acoustic frames.
    EncodedFrames,
    /// Actual tokenizer labels.
    TextLabels,
}

/// Preparation cannot produce an executable model request.
#[derive(
    Debug,
    Clone,
    PartialEq,
    Eq,
    Serialize,
    Deserialize,
    schemars::JsonSchema,
    utoipa::ToSchema,
    thiserror::Error,
)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum FaPreparationRefusalV2 {
    /// A constraint from the loaded model's configuration or tokenizer.
    #[error("loaded FA model {engine} admits at most {maximum} {unit:?}; request needs {provided}")]
    ModelConstraint {
        /// Units of both populations.
        unit: FaConstraintUnitV2,
        /// Requested population.
        provided: u64,
        /// The actual model bound.
        maximum: u64,
        /// Which loaded model imposed it.
        #[schema(value_type = String)]
        engine: ReportedEngineName,
    },
    /// Input below the minimum extent required by the loaded model.
    #[error(
        "loaded FA model {engine} needs at least {minimum} {unit:?}; input supplies {provided}"
    )]
    InsufficientInput {
        /// Units of both populations.
        unit: FaConstraintUnitV2,
        /// Actual input population.
        provided: u64,
        /// The model's required minimum.
        minimum: u64,
        /// The loaded model imposing the constraint.
        #[schema(value_type = String)]
        engine: ReportedEngineName,
    },
    /// Allocation projection was unavailable; no guessed footprint substitutes.
    #[error("FA allocation projection unavailable: {message}")]
    ProjectionUnavailable {
        /// The original qualified diagnostic.
        message: String,
    },
}

/// Exact loaded-model or physical-host constraint retained for unavailable
/// source words. These quantities describe a request, not a duration policy.
#[derive(
    Debug,
    Clone,
    PartialEq,
    Eq,
    Serialize,
    Deserialize,
    schemars::JsonSchema,
    utoipa::ToSchema,
    thiserror::Error,
)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum FaResourceConstraintV2 {
    /// Decoder measurement, before a model request could be prepared.
    #[error("prepared FA audio has no whole samples: {shortfall}")]
    EmptyDecodedAudio {
        /// Preserve no bytes versus a partial sample; infer no acoustic cause.
        shortfall: DecodedAudioShortfallV2,
    },
    /// The loaded backend's original typed preparation refusal.
    #[error("{cause}")]
    Model {
        /// The original quantities, model identity and diagnostic.
        cause: FaPreparationRefusalV2,
    },
    /// Full request demand exceeds total usable physical host capacity.
    #[error(
        "FA model {engine} needs {required_mb} MiB including its request; host has {total_mb} MiB with {reserve_mb} MiB reserved"
    )]
    HostMemory {
        /// The loaded model that projected this request.
        #[schema(value_type = String)]
        engine: ReportedEngineName,
        /// Model, request workspace, activation margin and worker baseline.
        required_mb: u64,
        /// Capacity unavailable to model requests by host policy.
        reserve_mb: u64,
        /// Physical capacity, independent of transient memory pressure.
        total_mb: u64,
    },
}

/// Wire representation of an admitted empty decoder observation. It grants no
/// audio or inference capability and never claims the source was past EOF.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Serialize,
    Deserialize,
    schemars::JsonSchema,
    utoipa::ToSchema,
    thiserror::Error,
)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum DecodedAudioShortfallV2 {
    /// The decoder produced zero bytes.
    #[error("the decoder produced no bytes")]
    NoBytes,
    /// Nonzero bytes did not complete one sample of the selected encoding.
    #[error("the decoder produced {decoded_bytes} bytes, less than one {sample_bytes}-byte sample")]
    PartialSample {
        /// Measured nonzero output bytes.
        #[schema(value_type = u64, minimum = 1)]
        decoded_bytes: std::num::NonZeroU64,
        /// Bytes per whole sample in the actual decoder encoding.
        #[schema(value_type = u64, minimum = 1)]
        sample_bytes: std::num::NonZeroU64,
    },
}

/// Preparation completion, separate from an inference result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum FaPreparationOutcomeV2 {
    /// The report still needs a consuming admission transition on the host.
    Prepared { plan: FaPreparationPlanV2 },
    /// No executable request was produced.
    Refused { cause: FaPreparationRefusalV2 },
}

/// Separate preparation completion from the outer task-result discriminator.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FaPreparationResultV2 {
    /// A bound plan or the exact preparation refusal.
    pub outcome: FaPreparationOutcomeV2,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fa_resource_quantities_refuse_zero_and_do_not_share_a_dimension()
    -> Result<(), serde_json::Error> {
        assert!(serde_json::from_str::<FaWorkspaceBytesV2>("0").is_err());
        assert!(serde_json::from_str::<FaModelBytesV2>("0").is_err());
        assert!(serde_json::from_str::<FaSampleCountV2>("0").is_err());
        let workspace: FaWorkspaceBytesV2 = serde_json::from_str("4096")?;
        assert_eq!(workspace.get(), 4096);
        Ok(())
    }

    #[test]
    fn fa_resource_absence_is_explicit_and_partial_real_demands_are_refused()
    -> Result<(), serde_json::Error> {
        assert_eq!(
            serde_json::from_str::<FaRequestMemoryV2>(r#"{"kind":"no_model"}"#)?,
            FaRequestMemoryV2::NoModel
        );
        assert!(
            serde_json::from_str::<FaRequestMemoryV2>(r#"{"kind":"projected","workspace":1}"#)
                .is_err()
        );
        assert!(serde_json::from_str::<FaRequestMemoryV2>(r#"{"kind":"unknown"}"#).is_err());
        Ok(())
    }
}
