//! Shared stage identities and observation for consuming pipeline transitions.

use std::fmt;
use std::future::Future;
use std::time::Instant;

use tracing::info;

use crate::error::ServerError;

/// Identifiers for internal pipeline stages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum StageId {
    /// Parse input content.
    Parse,
    /// Run pre-validation.
    PreValidate,
    /// Clear existing derived tiers or annotations.
    ClearExisting,
    /// Extract worker payloads.
    CollectPayloads,
    /// Run worker inference.
    Infer,
    /// Apply inference results to the document.
    ApplyResults,
    /// Run post-validation.
    PostValidate,
    /// Run ASR inference.
    AsrInfer,
    /// Run dedicated speaker diarization when requested.
    SpeakerDiarization,
    /// Convert ASR output into utterances.
    AsrPostprocess,
    /// Build CHAT from utterances.
    BuildChat,
    /// Optional utterance segmentation pass.
    OptionalUtseg,
    /// Optional morphosyntax pass.
    OptionalMorphosyntax,
    /// Finalize the output text.
    Serialize,
}

impl StageId {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Parse => "parse",
            Self::PreValidate => "pre_validate",
            Self::ClearExisting => "clear_existing",
            Self::CollectPayloads => "collect_payloads",
            Self::Infer => "infer",
            Self::ApplyResults => "apply_results",
            Self::PostValidate => "post_validate",
            Self::AsrInfer => "asr_infer",
            Self::SpeakerDiarization => "speaker_diarization",
            Self::AsrPostprocess => "asr_postprocess",
            Self::BuildChat => "build_chat",
            Self::OptionalUtseg => "optional_utseg",
            Self::OptionalMorphosyntax => "optional_morphosyntax",
            Self::Serialize => "serialize",
        }
    }
}

impl fmt::Display for StageId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Observe a consuming transition without erasing its output type or boxing it.
pub(crate) async fn observe_stage<T>(
    command: &'static str,
    stage: StageId,
    transition: impl Future<Output = Result<T, ServerError>>,
) -> Result<T, ServerError> {
    let started = Instant::now();
    info!(command, stage = %stage, "Starting pipeline stage");
    let output = transition.await?;
    info!(command, stage = %stage,
        duration_ms = started.elapsed().as_millis() as u64,
        "Completed pipeline stage");
    Ok(output)
}
