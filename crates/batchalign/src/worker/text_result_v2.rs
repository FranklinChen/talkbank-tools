//! Rust-side adapters for batched text worker-protocol V2 results.
//!
//! The text-task orchestrators still own cache policy, postprocessing, and CHAT
//! mutation in Rust. This module keeps the worker boundary typed by parsing the
//! batched `execute_v2` result variants before task-local orchestration logic
//! reconstructs established Rust domains.

use crate::types::worker_v2::{
    CorefResultV2, ExecuteResponseV2, MorphosyntaxResultV2, TaskResultV2, TranslationResultV2,
    UtsegResultV2,
};
use crate::worker::execute_result_v2::{require_success_result, take_success_result};

/// Take the typed batch result out of one V2 morphosyntax execute response.
///
/// By value: each item's model identity and raw sentences move into the
/// admitted response rather than being cloned out of a borrowed payload.
pub fn parse_morphosyntax_result_v2(
    response: ExecuteResponseV2,
) -> Result<MorphosyntaxResultV2, String> {
    match take_success_result(response, "morphosyntax")? {
        TaskResultV2::MorphosyntaxResult(result) => Ok(result),
        _ => Err("worker protocol V2 morphosyntax response returned the wrong result type".into()),
    }
}

/// Parse one V2 utterance-segmentation execute response into the typed batch
/// result.
pub fn parse_utseg_result_v2(response: &ExecuteResponseV2) -> Result<&UtsegResultV2, String> {
    match require_success_result(response, "utseg")? {
        TaskResultV2::UtsegResult(result) => Ok(result),
        _ => Err("worker protocol V2 utseg response returned the wrong result type".into()),
    }
}

/// Take the typed batch result out of one V2 translation execute response.
/// By value, so each item's translation and engine name move rather than clone.
pub fn parse_translate_result_v2(
    response: ExecuteResponseV2,
) -> Result<TranslationResultV2, String> {
    match take_success_result(response, "translate")? {
        TaskResultV2::TranslationResult(result) => Ok(result),
        _ => Err("worker protocol V2 translate response returned the wrong result type".into()),
    }
}

/// Take the typed batch result out of one V2 coreference execute response.
/// By value, so each item's annotations and engine name move rather than clone.
pub fn parse_coref_result_v2(response: ExecuteResponseV2) -> Result<CorefResultV2, String> {
    match take_success_result(response, "coref")? {
        TaskResultV2::CorefResult(result) => Ok(result),
        _ => Err("worker protocol V2 coref response returned the wrong result type".into()),
    }
}
