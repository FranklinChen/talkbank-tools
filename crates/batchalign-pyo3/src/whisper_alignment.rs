//! Shared lexical DTW boundary for Whisper generation and forced alignment.

use batchalign_fa_core::{CostMatrix, LexicalFrameAlignment};
use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::{exceptions::PyValueError, prelude::*};

/// Return one frame per label using open-begin, fixed-end alignment.
#[pyfunction]
pub(crate) fn whisper_lexical_frame_indices(
    py: Python<'_>,
    costs: PyReadonlyArray2<'_, f32>,
) -> PyResult<Vec<usize>> {
    let shape = costs.shape();
    let matrix = CostMatrix::new(
        shape[0],
        shape[1],
        costs.as_array().iter().copied().collect(),
    )
    .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(py.detach(move || LexicalFrameAlignment::align(&matrix).row_entries().to_vec()))
}
