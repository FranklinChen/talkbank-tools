#![warn(missing_docs)]
//! Numeric core for Whisper alignment: median filter, column standardization,
//! and DTW. The Python ASR and FA consumers share the lexical boundary through
//! PyO3; this leaf crate needs no model or tensor runtime.
//!
//! Port of the numeric primitives the Python FA path uses
//! from `transformers.models.whisper.generation_whisper`
//! (`_median_filter`, `_dynamic_time_warping`), so the Rust FA path can
//! apply the same alignment stages: teacher-forced forward
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
    /// A cost cannot participate in a finite shortest-path comparison.
    #[error("cost matrix contains a non-finite value at offset {0}")]
    NonFinite(usize),
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
        if rows.checked_mul(cols) != Some(values.len()) {
            return Err(FaDtwError::ShapeMismatch {
                rows,
                cols,
                len: values.len(),
            });
        }
        if let Some(index) = values.iter().position(|value| !value.is_finite()) {
            return Err(FaDtwError::NonFinite(index));
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
/// the closed-begin variant (moves: diagonal, down, right). Ties prefer
/// the diagonal, then the text axis. Accumulation uses f64 for finite f32 costs.
pub fn dynamic_time_warping(matrix: &CostMatrix) -> Result<Vec<PathPoint>, FaDtwError> {
    Ok(warp(matrix, BeginBoundary::Closed))
}

#[derive(Clone, Copy)]
enum BeginBoundary {
    Closed,
    Open,
}

/// Complete, ordered row-entry frames produced by open-begin, fixed-end DTW.
/// Every input label is represented; only an audio prefix may be unassigned.
/// This proves path geometry, not acoustic correctness or a unique occurrence.
#[derive(Debug)]
pub struct LexicalFrameAlignment {
    row_entries: Vec<usize>,
}

impl LexicalFrameAlignment {
    /// Align every row while allowing the first row to start after frame zero.
    pub fn align(matrix: &CostMatrix) -> Self {
        let path = warp(matrix, BeginBoundary::Open);
        let mut row_entries = Vec::with_capacity(matrix.rows);
        for point in path {
            if point.text_idx == row_entries.len() {
                row_entries.push(point.time_idx);
            }
        }
        Self { row_entries }
    }

    /// One bounded, monotone frame index per input row, in input row order.
    pub fn row_entries(&self) -> &[usize] {
        &self.row_entries
    }
}

fn warp(matrix: &CostMatrix, begin: BeginBoundary) -> Vec<PathPoint> {
    // Shape validity is a `CostMatrix::new` invariant; fields are private.
    let (n, m) = (matrix.rows, matrix.cols);
    // cost[(i, j)] over a (n+1) x (m+1) grid with an infinite border,
    // trace stores which move produced each cell (0 diag, 1 down, 2 right).
    let width = m + 1;
    // f64 accumulation keeps finite f32 input costs finite along long paths.
    let mut cost = vec![f64::INFINITY; (n + 1) * width];
    let mut trace = vec![u8::MAX; (n + 1) * width];
    cost[0] = 0.0;
    if matches!(begin, BeginBoundary::Open) {
        cost[..width].fill(0.0);
    }
    for i in 1..=n {
        for j in 1..=m {
            let c0 = cost[(i - 1) * width + (j - 1)];
            let c1 = cost[(i - 1) * width + j];
            let c2 = cost[i * width + (j - 1)];
            // Deterministic minimum: ties prefer diagonal, then text axis.
            let (best, mv) = if c0 <= c1 && c0 <= c2 {
                (c0, 0u8)
            } else if c1 <= c2 {
                (c1, 1u8)
            } else {
                (c2, 2u8)
            };
            cost[i * width + j] = f64::from(matrix.at(i - 1, j - 1)) + best;
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
    path
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
    fn open_begin_skips_prefix_but_covers_all_labels_and_fixed_end() {
        let matrix = CostMatrix::new(2, 4, vec![9., 9., -4., 9., 9., 9., 9., -4.]).unwrap();
        let path = warp(&matrix, BeginBoundary::Open);
        assert_eq!(LexicalFrameAlignment::align(&matrix).row_entries(), &[2, 3]);
        assert_eq!(
            path.last(),
            Some(&PathPoint {
                text_idx: 1,
                time_idx: 3
            })
        );
        assert_eq!(dynamic_time_warping(&matrix).unwrap()[0].time_idx, 0);
    }

    #[test]
    fn open_begin_agrees_with_exhaustive_small_signed_cost_oracle() {
        // Independent enumeration of every legal start and monotone path.
        fn cheapest(matrix: &CostMatrix, row: usize, col: usize) -> f64 {
            let value = f64::from(matrix.at(row, col));
            if row + 1 == matrix.rows && col + 1 == matrix.cols {
                return value;
            }
            let mut suffix = f64::INFINITY;
            if row + 1 < matrix.rows {
                suffix = suffix.min(cheapest(matrix, row + 1, col));
            }
            if col + 1 < matrix.cols {
                suffix = suffix.min(cheapest(matrix, row, col + 1));
            }
            if row + 1 < matrix.rows && col + 1 < matrix.cols {
                suffix = suffix.min(cheapest(matrix, row + 1, col + 1));
            }
            value + suffix
        }
        for rows in 1..=3 {
            for cols in 1..=3 {
                for bits in 0..(1usize << (rows * cols)) {
                    let values = (0..rows * cols)
                        .map(|index| if bits & (1 << index) == 0 { -1. } else { 2. })
                        .collect();
                    let matrix = CostMatrix::new(rows, cols, values).unwrap();
                    let path = warp(&matrix, BeginBoundary::Open);
                    let observed: f64 = path
                        .iter()
                        .map(|p| f64::from(matrix.at(p.text_idx, p.time_idx)))
                        .sum();
                    let expected = (0..cols)
                        .map(|col| cheapest(&matrix, 0, col))
                        .fold(f64::INFINITY, f64::min);
                    assert_eq!(observed, expected);
                    let entries = LexicalFrameAlignment::align(&matrix);
                    assert_eq!(entries.row_entries().len(), rows);
                    assert!(entries.row_entries().windows(2).all(|p| p[0] <= p[1]));
                    assert!(entries.row_entries().iter().all(|&col| col < cols));
                }
            }
        }
    }

    #[test]
    fn repeated_labels_and_zero_cost_ties_do_not_drop_rows() {
        let matrix = CostMatrix::new(4, 2, vec![0.; 8]).unwrap();
        assert_eq!(
            LexicalFrameAlignment::align(&matrix).row_entries(),
            &[0, 0, 0, 1]
        );
        // Negative costs may still favor assigning the prefix. Open-begin is
        // permission to skip, never evidence that a frame is physically silent.
        let matrix = CostMatrix::new(1, 4, vec![-1.; 4]).unwrap();
        assert_eq!(LexicalFrameAlignment::align(&matrix).row_entries(), &[0]);
    }

    #[test]
    fn rejects_nonfinite_and_overflowing_shapes() {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(
                CostMatrix::new(1, 1, vec![value]).unwrap_err(),
                FaDtwError::NonFinite(0)
            );
        }
        assert!(matches!(
            CostMatrix::new(usize::MAX, 2, vec![]),
            Err(FaDtwError::ShapeMismatch { .. })
        ));
    }

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
