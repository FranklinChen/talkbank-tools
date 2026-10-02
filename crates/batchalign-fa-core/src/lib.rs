#![warn(missing_docs)]
//! Numeric core for Whisper forced alignment: median filter, column
//! standardization, and DTW. A LEAF crate (thiserror only) so both the
//! batchalign server crate and the whisper pilot depend on the same
//! implementation without either depending on the other's stack.
//!
//! Faithful port of the two numeric primitives the Python FA path uses
//! from `transformers.models.whisper.generation_whisper`
//! (`_median_filter`, `_dynamic_time_warping`), so the Rust FA path can
//! reproduce the HF alignment algorithm exactly: teacher-forced forward
//! pass -> alignment-head cross-attentions -> per-head standardize ->
//! median filter -> head-mean cost matrix -> DTW -> token jump times at
//! 20 ms per audio frame.
//!
//! Kept dependency-free (plain `Vec<f32>` matrices) so the algorithm is
//! testable without any model or tensor runtime; a model integration layers
//! on top (the candle one lived in the retired Whisper pilot crate).

/// A dense row-major matrix of attention costs: `rows` = text tokens,
/// `cols` = audio frames.
#[derive(Debug, Clone)]
pub struct CostMatrix {
    rows: usize,
    cols: usize,
    /// Row-major; `len == rows * cols`, enforced by the constructor.
    values: Vec<f32>,
}

/// Everything that can go wrong in the numeric FA core.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum FaDtwError {
    /// Matrix dimensions do not match the value buffer.
    #[error("cost matrix shape mismatch: {rows}x{cols} != {len} values")]
    ShapeMismatch {
        /// Declared row count.
        rows: usize,
        /// Declared column count.
        cols: usize,
        /// Actual buffer length.
        len: usize,
    },
    /// The matrix is empty on at least one axis; DTW is undefined.
    #[error("cost matrix is empty ({rows}x{cols})")]
    Empty {
        /// Declared row count.
        rows: usize,
        /// Declared column count.
        cols: usize,
    },
    /// Median filter width must be odd (HF asserts the same).
    #[error("median filter width {0} is not odd")]
    EvenFilterWidth(usize),
}

impl CostMatrix {
    /// Construct with shape validation.
    pub fn new(rows: usize, cols: usize, values: Vec<f32>) -> Result<Self, FaDtwError> {
        if rows == 0 || cols == 0 {
            return Err(FaDtwError::Empty { rows, cols });
        }
        if values.len() != rows * cols {
            return Err(FaDtwError::ShapeMismatch {
                rows,
                cols,
                len: values.len(),
            });
        }
        Ok(Self { rows, cols, values })
    }

    fn at(&self, r: usize, c: usize) -> f32 {
        self.values[r * self.cols + c]
    }

    /// Row-major view of the values.
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// Consume into the row-major value buffer.
    pub fn into_values(self) -> Vec<f32> {
        self.values
    }
}

/// One step of the DTW path: which text row aligns to which frame column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PathPoint {
    /// Text-token index.
    pub text_idx: usize,
    /// Audio-frame index.
    pub time_idx: usize,
}

/// Monotone alignment path through the cost matrix, cheapest-first, as
/// `transformers`' `_dynamic_time_warping` computes it (moves: diagonal,
/// down, right; ties prefer the diagonal, then the text axis, matching
/// the `argmin` order of the reference implementation).
pub fn dynamic_time_warping(matrix: &CostMatrix) -> Result<Vec<PathPoint>, FaDtwError> {
    // Shape validity is a `CostMatrix::new` invariant; fields are private.
    let (n, m) = (matrix.rows, matrix.cols);
    // cost[(i, j)] over a (n+1) x (m+1) grid with an infinite border,
    // trace stores which move produced each cell (0 diag, 1 down, 2 right).
    let width = m + 1;
    let mut cost = vec![f32::INFINITY; (n + 1) * width];
    let mut trace = vec![u8::MAX; (n + 1) * width];
    cost[0] = 0.0;
    for i in 1..=n {
        for j in 1..=m {
            let c0 = cost[(i - 1) * width + (j - 1)];
            let c1 = cost[(i - 1) * width + j];
            let c2 = cost[i * width + (j - 1)];
            // Reference tie-break: argmin([diag, down, right]).
            let (best, mv) = if c0 <= c1 && c0 <= c2 {
                (c0, 0u8)
            } else if c1 <= c2 {
                (c1, 1u8)
            } else {
                (c2, 2u8)
            };
            cost[i * width + j] = matrix.at(i - 1, j - 1) + best;
            trace[i * width + j] = mv;
        }
    }
    // Backtrace from (n, m).
    let mut path = Vec::with_capacity(n + m);
    let (mut i, mut j) = (n, m);
    while i > 0 && j > 0 {
        path.push(PathPoint {
            text_idx: i - 1,
            time_idx: j - 1,
        });
        match trace[i * width + j] {
            0 => {
                i -= 1;
                j -= 1;
            }
            1 => i -= 1,
            _ => j -= 1,
        }
    }
    path.reverse();
    Ok(path)
}

/// In-place median filter along the frame (column) axis with reflect
/// padding, matching `transformers`' `_median_filter` semantics for a
/// 2-D row-major matrix (each row filtered independently).
pub fn median_filter_rows(matrix: &mut CostMatrix, width: usize) -> Result<(), FaDtwError> {
    if width.is_multiple_of(2) {
        return Err(FaDtwError::EvenFilterWidth(width));
    }
    if width <= 1 || matrix.cols == 1 {
        return Ok(());
    }
    // Reference behavior: if the axis is shorter than the filter width,
    // the effective width shrinks to the largest odd size that fits.
    let width = width.min(if matrix.cols.is_multiple_of(2) {
        matrix.cols - 1
    } else {
        matrix.cols
    });
    let half = width / 2;
    let cols = matrix.cols;
    let mut window = vec![0.0f32; width];
    // One scratch buffer reused across rows (a per-row alloc costs ~1.2k
    // allocations per FA call at production head counts).
    let mut original = vec![0.0f32; cols];
    for r in 0..matrix.rows {
        let row_start = r * cols;
        original.copy_from_slice(&matrix.values[row_start..row_start + cols]);
        for c in 0..cols {
            for (k, slot) in window.iter_mut().enumerate() {
                // Reflect padding (PyTorch "reflect": mirror without
                // repeating the edge sample).
                let idx = c as isize + k as isize - half as isize;
                let idx = if idx < 0 {
                    (-idx) as usize
                } else if idx as usize >= cols {
                    2 * (cols - 1) - idx as usize
                } else {
                    idx as usize
                };
                *slot = original[idx];
            }
            window.sort_by(|a, b| a.total_cmp(b));
            matrix.values[row_start + c] = window[half];
        }
    }
    Ok(())
}

/// Standardize each COLUMN of a row-major matrix in place: for every
/// column, subtract the column mean and divide by the (population)
/// standard deviation. This is step 3 of the HF forced-alignment
/// algorithm (`torch.std_mean(weights, dim=-2, unbiased=False)` over
/// the token axis), shared here so the pilot and production paths use
/// one implementation. Single row-major accumulation pass, then one
/// normalize pass (cache-friendly on the row-major buffer).
pub fn standardize_columns(matrix: &mut CostMatrix) {
    let (rows, cols) = (matrix.rows, matrix.cols);
    let mut sum = vec![0.0f64; cols];
    let mut sumsq = vec![0.0f64; cols];
    for r in 0..rows {
        let row = &matrix.values[r * cols..(r + 1) * cols];
        for (c, v) in row.iter().enumerate() {
            let v = f64::from(*v);
            sum[c] += v;
            sumsq[c] += v * v;
        }
    }
    let n = rows as f64;
    let mut mean = sum;
    let mut inv_std = sumsq;
    for c in 0..cols {
        mean[c] /= n;
        let var = (inv_std[c] / n - mean[c] * mean[c]).max(0.0);
        inv_std[c] = 1.0 / var.sqrt().max(f64::EPSILON);
    }
    for r in 0..rows {
        let row = &mut matrix.values[r * cols..(r + 1) * cols];
        for (c, v) in row.iter_mut().enumerate() {
            *v = ((f64::from(*v) - mean[c]) * inv_std[c]) as f32;
        }
    }
}

/// Token jump times: for each text row, the first frame where the DTW
/// path enters that row, in seconds at the Whisper frame rate (20 ms).
pub fn token_jump_times_s(path: &[PathPoint]) -> Vec<f64> {
    const FRAME_S: f64 = 0.02;
    let mut out = Vec::new();
    let mut prev_text = usize::MAX;
    for p in path {
        if p.text_idx != prev_text {
            out.push(p.time_idx as f64 * FRAME_S);
            prev_text = p.text_idx;
        }
    }
    out
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn dtw_identity_diagonal() {
        // A cost matrix favoring the diagonal must yield the diagonal path.
        let m = CostMatrix::new(3, 3, vec![0.0, 9.0, 9.0, 9.0, 0.0, 9.0, 9.0, 9.0, 0.0]).unwrap();
        let path = dynamic_time_warping(&m).unwrap();
        assert_eq!(
            path,
            vec![
                PathPoint {
                    text_idx: 0,
                    time_idx: 0
                },
                PathPoint {
                    text_idx: 1,
                    time_idx: 1
                },
                PathPoint {
                    text_idx: 2,
                    time_idx: 2
                },
            ]
        );
    }

    #[test]
    fn dtw_wide_matrix_stretches_along_frames() {
        // One token over four frames: the path stays on the single row.
        let m = CostMatrix::new(1, 4, vec![0.1, 0.1, 0.1, 0.1]).unwrap();
        let path = dynamic_time_warping(&m).unwrap();
        assert_eq!(path.len(), 4);
        assert!(path.iter().all(|p| p.text_idx == 0));
        assert_eq!(token_jump_times_s(&path), vec![0.0]);
    }

    #[test]
    fn dtw_rejects_empty() {
        assert_eq!(
            CostMatrix::new(0, 3, vec![]).unwrap_err(),
            FaDtwError::Empty { rows: 0, cols: 3 }
        );
    }

    #[test]
    fn jump_times_are_row_entry_frames() {
        let path = vec![
            PathPoint {
                text_idx: 0,
                time_idx: 0,
            },
            PathPoint {
                text_idx: 0,
                time_idx: 1,
            },
            PathPoint {
                text_idx: 1,
                time_idx: 2,
            },
            PathPoint {
                text_idx: 2,
                time_idx: 2,
            },
            PathPoint {
                text_idx: 2,
                time_idx: 3,
            },
        ];
        let times = token_jump_times_s(&path);
        assert_eq!(times, vec![0.0, 0.04, 0.04]);
    }

    #[test]
    fn median_filter_matches_reference_semantics() {
        // Reference: scipy-style median with reflect padding, width 3.
        // Row [1, 5, 2, 4] -> [padded 5,1,5,2,4,2] windows:
        //   [5,1,5]->5, [1,5,2]->2, [5,2,4]->4, [2,4,2]->2
        let mut m = CostMatrix::new(1, 4, vec![1.0, 5.0, 2.0, 4.0]).unwrap();
        median_filter_rows(&mut m, 3).unwrap();
        assert_eq!(m.values, vec![5.0, 2.0, 4.0, 2.0]);
    }

    #[test]
    fn median_filter_rejects_even_width() {
        let mut m = CostMatrix::new(1, 4, vec![1.0, 5.0, 2.0, 4.0]).unwrap();
        assert_eq!(
            median_filter_rows(&mut m, 4).unwrap_err(),
            FaDtwError::EvenFilterWidth(4)
        );
    }

    #[test]
    fn standardize_columns_zero_mean_unit_variance() {
        // Column [1, 3] -> mean 2, std 1 -> [-1, 1]; column [5, 5] is
        // constant -> stays 0 (guarded divide).
        let mut m = CostMatrix::new(2, 2, vec![1.0, 5.0, 3.0, 5.0]).unwrap();
        standardize_columns(&mut m);
        assert_eq!(m.values, vec![-1.0, 0.0, 1.0, 0.0]);
    }
}
