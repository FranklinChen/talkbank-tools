//! Rust-side adapters for batched text worker-protocol V2 results.
//!
//! The text-task orchestrators still own cache policy, postprocessing, and CHAT
//! mutation in Rust. This module keeps the worker boundary typed by parsing the
//! batched `execute_v2` result variants before task-local orchestration logic
//! reconstructs established Rust domains.
//!
//! A refusal keeps the worker's protocol code ([`TextResultRefusal::Failed`])
//! until [`TextResultRefusal::into_server_error`] decides its category from
//! that code, so a worker with no pipeline for a language, a gated model and
//! a model crash stay three different failures rather than one string each
//! caller labelled for itself.

use crate::error::ServerError;
use crate::types::worker_v2::{
    CorefResultV2, ExecuteResponseV2, MorphosyntaxResultV2, TaskResultV2, TranslationResultV2,
    UtsegResultV2, WordSegmentationResultV2,
};
use crate::worker::error::WorkerError;
use crate::worker::execute_result_v2::{
    ExecuteFailureRead, ModelLocality, require_success_result, take_success_result,
};

/// Why a text execute response is not the result its task asked for.
#[derive(Debug, thiserror::Error)]
pub enum TextResultRefusal {
    /// The worker answered with a failure, which keeps its protocol code.
    #[error(transparent)]
    Failed(#[from] ExecuteFailureRead),
    /// A successful response carried another task's result: the worker broke
    /// the protocol.
    #[error("worker protocol V2 {task} response returned the wrong result type")]
    WrongResult {
        /// The task the response was read for.
        task: &'static str,
    },
}

impl TextResultRefusal {
    /// The server error this refusal is: a worker failure by its protocol
    /// code and where the task's model runs, a wrong result as a protocol
    /// breach.
    pub(crate) fn into_server_error(self, locality: ModelLocality) -> ServerError {
        match self {
            Self::Failed(failure) => {
                failure.into_server_error(locality, crate::error::BilledProduct::NotNamed)
            }
            wrong @ Self::WrongResult { .. } => {
                ServerError::Worker(WorkerError::Protocol(wrong.to_string()))
            }
        }
    }
}

/// Take the typed batch result out of one V2 morphosyntax execute response.
///
/// By value: each item's model identity and raw sentences move into the
/// admitted response rather than being cloned out of a borrowed payload.
pub fn parse_morphosyntax_result_v2(
    response: ExecuteResponseV2,
) -> Result<MorphosyntaxResultV2, TextResultRefusal> {
    match take_success_result(response, "morphosyntax")? {
        TaskResultV2::MorphosyntaxResult(result) => Ok(result),
        _ => Err(TextResultRefusal::WrongResult {
            task: "morphosyntax",
        }),
    }
}

/// Parse one V2 utterance-segmentation execute response into the typed batch
/// result.
pub fn parse_utseg_result_v2(
    response: &ExecuteResponseV2,
) -> Result<&UtsegResultV2, TextResultRefusal> {
    match require_success_result(response, "utseg")? {
        TaskResultV2::UtsegResult(result) => Ok(result),
        _ => Err(TextResultRefusal::WrongResult { task: "utseg" }),
    }
}

/// Take the typed batch result out of one V2 translation execute response.
/// By value, so each item's translation and engine name move rather than clone.
pub fn parse_translate_result_v2(
    response: ExecuteResponseV2,
) -> Result<TranslationResultV2, TextResultRefusal> {
    match take_success_result(response, "translate")? {
        TaskResultV2::TranslationResult(result) => Ok(result),
        _ => Err(TextResultRefusal::WrongResult { task: "translate" }),
    }
}

/// Take the typed batch result out of one V2 coreference execute response.
/// By value, so each item's annotations and engine name move rather than clone.
pub fn parse_coref_result_v2(
    response: ExecuteResponseV2,
) -> Result<CorefResultV2, TextResultRefusal> {
    match take_success_result(response, "coref")? {
        TaskResultV2::CorefResult(result) => Ok(result),
        _ => Err(TextResultRefusal::WrongResult { task: "coref" }),
    }
}

/// Take the typed result out of one V2 word-segmentation execute response.
/// By value, so the token lists and the segmenter's name move rather than clone.
pub fn parse_word_segmentation_result_v2(
    response: ExecuteResponseV2,
) -> Result<WordSegmentationResultV2, TextResultRefusal> {
    match take_success_result(response, "word segmentation")? {
        TaskResultV2::WordSegmentationResult(result) => Ok(result),
        _ => Err(TextResultRefusal::WrongResult {
            task: "word segmentation",
        }),
    }
}

#[cfg(test)]
mod tests;
