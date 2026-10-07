//! Shared stage identities and observation for consuming pipeline transitions.

use std::fmt;
use std::time::Instant;

use tracing::info;

use crate::error::ServerError;

/// Identifiers for internal pipeline stages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum StageId {
    /// Parse input content.
    Parse,
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

/// One pipeline stage's transition, owned on the heap by whoever produces it.
///
/// A stage future passed BY VALUE into an `async fn` is stored twice in the
/// caller's state machine (rust-lang/rust#62958), and wrapping stages in
/// observers nests that doubling at every level. Measured on 2026-10-06 with
/// `-Zprint-type-sizes`: the morphosyntax stage (45.6 KB) became 91 KB inside
/// `observe_stage`, 137 KB inside the transcribe progress wrapper, and the
/// whole transcribe pipeline about 140 KB, copied onto the stack at every poll
/// layer until a production tokio worker overflowed its 2 MiB stack. Requiring
/// this type makes a stage cost one pointer in its caller, and no caller can
/// pass an inline future again: the compiler refuses it. The general rule
/// lives in [`crate::owned_future`].
pub(crate) type StageFuture<'a, T> = crate::owned_future::OwnedFuture<'a, Result<T, ServerError>>;

/// Observe a consuming transition without erasing its output type.
pub(crate) async fn observe_stage<T>(
    command: &'static str,
    stage: StageId,
    transition: StageFuture<'_, T>,
) -> Result<T, ServerError> {
    let started = Instant::now();
    info!(command, stage = %stage, "Starting pipeline stage");
    let output = transition.await?;
    info!(command, stage = %stage,
        duration_ms = started.elapsed().as_millis() as u64,
        "Completed pipeline stage");
    Ok(output)
}
