//! Rust-owned normalization for worker-protocol V2 text-task batch results.
//!
//! **See also:** [INTERFACE_MAP.md](../../../INTERFACE_MAP.md) section "7. Text Task Result Normalization" for:
//! - Python caller: `batchalign/worker/_text_v2.py`
//! - Full Rust/Python responsibility split and input/output contracts.

use batchalign_types::worker::{BatchInferResponse, InferResponse, ItemOutcome};
use batchalign_types::worker_v2::{
    CorefItemResultV2, CorefResultV2, MorphosyntaxItemResultV2, MorphosyntaxResultV2,
    TranslationItemResultV2, TranslationResultV2, UtsegItemResultV2, UtsegResultV2,
};
use pyo3::prelude::*;
use serde::de::DeserializeOwned;

/// Why a host response could not be normalized into a typed V2 payload.
///
/// A plain message: the CATEGORY is always the same (the host returned a
/// shape the contract does not allow), and the two consumers wrap it
/// differently at their own boundaries (the pyfunction raises it as an
/// internal error; the Rust text executor folds it into
/// `runtime_failure` with the established "invalid <task> host output"
/// prefix the Python test matrix asserts on).
pub(crate) struct TextResultShapeError(pub(crate) String);

fn normalize_result_count<'a>(
    response: &'a BatchInferResponse,
    expected_count: usize,
    task: &str,
) -> Result<&'a [InferResponse], TextResultShapeError> {
    let actual_count = response.results.len();
    if actual_count != expected_count {
        return Err(TextResultShapeError(format!(
            "worker protocol V2 {task} host returned {actual_count} items, expected {expected_count}"
        )));
    }
    Ok(response.results.as_slice())
}

/// A tagged per-item result whose failure variant the bridge can build.
///
/// The morphosyntax, utseg, translation and coreference hosts return each item's
/// `result` already in its V2 tagged shape (`{"kind": "analyzed", ...}`), so
/// the bridge parses it through the canonical Rust wire type rather than
/// re-reading fields. Anything that does not parse becomes THAT item's
/// failure, the same per-item attribution the server applies, instead of
/// failing every other item in the batch.
trait HostItemResult: DeserializeOwned {
    /// The item-level failure carrying `error`.
    fn failed(error: String) -> Self;
}

impl HostItemResult for MorphosyntaxItemResultV2 {
    fn failed(error: String) -> Self {
        Self::Failed { error }
    }
}

impl HostItemResult for UtsegItemResultV2 {
    fn failed(error: String) -> Self {
        Self::Failed { error }
    }
}

impl HostItemResult for TranslationItemResultV2 {
    fn failed(error: String) -> Self {
        Self::Failed { error }
    }
}

impl HostItemResult for CorefItemResultV2 {
    fn failed(error: String) -> Self {
        Self::Failed { error }
    }
}

/// Normalize one host item into its tagged V2 result; a result is parsed
/// from the borrowed JSON without cloning it.
fn normalize_item<T: HostItemResult>(infer_result: &InferResponse, task: &str) -> T {
    match &infer_result.outcome {
        ItemOutcome::Failed { error } => T::failed(error.clone()),
        ItemOutcome::Produced { result } => T::deserialize(result.as_value())
            .unwrap_or_else(|error| T::failed(format!("invalid {task} host item: {error}"))),
    }
}

pub(crate) fn normalize_morphosyntax_result(
    response: &BatchInferResponse,
    expected_count: usize,
) -> Result<MorphosyntaxResultV2, TextResultShapeError> {
    Ok(MorphosyntaxResultV2 {
        items: normalize_result_count(response, expected_count, "morphosyntax")?
            .iter()
            .map(|infer_result| normalize_item(infer_result, "morphosyntax"))
            .collect(),
    })
}

pub(crate) fn normalize_utseg_result(
    response: &BatchInferResponse,
    expected_count: usize,
) -> Result<UtsegResultV2, TextResultShapeError> {
    Ok(UtsegResultV2 {
        items: normalize_result_count(response, expected_count, "utseg")?
            .iter()
            .map(|infer_result| normalize_item(infer_result, "utseg"))
            .collect(),
    })
}

pub(crate) fn normalize_translation_result(
    response: &BatchInferResponse,
    expected_count: usize,
) -> Result<TranslationResultV2, TextResultShapeError> {
    Ok(TranslationResultV2 {
        items: normalize_result_count(response, expected_count, "translate")?
            .iter()
            .map(|infer_result| normalize_item(infer_result, "translate"))
            .collect(),
    })
}

pub(crate) fn normalize_coref_result(
    response: &BatchInferResponse,
    expected_count: usize,
) -> Result<CorefResultV2, TextResultShapeError> {
    Ok(CorefResultV2 {
        items: normalize_result_count(response, expected_count, "coref")?
            .iter()
            .map(|infer_result| normalize_item(infer_result, "coref"))
            .collect(),
    })
}

// ---------------------------------------------------------------------------
// Token alignment (used by Python morphosyntax tokenizer realignment)
// ---------------------------------------------------------------------------

/// Align Stanza tokenizer output back to original CHAT words.
///
/// Returns a Python list: plain `str` for unchanged tokens,
/// `(str, bool)` tuples for MWT expansion hints.
#[pyfunction]
pub(crate) fn align_tokens(
    py: Python<'_>,
    original_words: Vec<String>,
    stanza_tokens: Vec<String>,
    alpha2: String,
) -> PyResult<Py<pyo3::types::PyList>> {
    use batchalign_transform::tokenizer_realign::{self, PatchedToken};
    use pyo3::types::{PyBool, PyList, PyString, PyTuple};

    let patched = py
        .detach(|| tokenizer_realign::align_tokens(&original_words, &stanza_tokens, &alpha2))
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;

    let result = PyList::empty(py);
    for tok in patched.tokens() {
        match tok {
            PatchedToken::Plain(s) => {
                result.append(PyString::new(py, s))?;
            }
            PatchedToken::Hint(s, expand) => {
                let s_any: Py<PyAny> = PyString::new(py, s).unbind().into_any();
                let b_any: Py<PyAny> = PyBool::new(py, *expand).to_owned().unbind().into_any();
                let tup = PyTuple::new(py, [s_any.bind(py), b_any.bind(py)])?;
                result.append(tup)?;
            }
            PatchedToken::FrenchElision(group) => {
                let surface = PyString::new(py, group.surface()).into_any();
                let components = PyList::new(py, group.components())?.into_any();
                result.append(PyTuple::new(py, [surface, components])?)?;
            }
        }
    }

    Ok(result.unbind())
}
