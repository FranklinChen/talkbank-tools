#![warn(missing_docs)]
//! Numeric core for Whisper lexical alignment. Python owns attention
//! normalization and filtering, then admits a cost matrix through PyO3.
//! This crate produces open-begin, fixed-end frame indices without a model
//! or tensor runtime. Frame-to-seconds conversion uses the caller's model geometry.

/// A dense row-major matrix of attention costs: `rows` = text tokens,
/// `cols` = audio frames.
#[derive(Debug)]
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

/// A path point can only be produced by the admitted matrix's traceback.
struct PathPoint {
    text_idx: usize,
    time_idx: usize,
}

#[derive(Clone, Copy)]
enum Step {
    Diagonal,
    Text,
    Frame,
}

/// Completed traceback retains the exact admitted matrix that produced it.
struct WarpedPath {
    matrix: CostMatrix,
    points: Vec<PathPoint>,
}

impl WarpedPath {
    /// Consume the completed path without repeating dynamic programming.
    fn into_alignment(self) -> LexicalFrameAlignment {
        let mut row_entries = Vec::with_capacity(self.matrix.rows);
        for point in self.points {
            if point.text_idx == row_entries.len() {
                row_entries.push(point.time_idx);
            }
        }
        LexicalFrameAlignment { row_entries }
    }
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
    pub fn align(matrix: CostMatrix) -> Self {
        warp(matrix).into_alignment()
    }

    /// One bounded, monotone frame index per input row, in input row order.
    pub fn row_entries(&self) -> &[usize] {
        &self.row_entries
    }

    /// Transfer the completed frame indices to the caller without copying them.
    pub fn into_row_entries(self) -> Vec<usize> {
        self.row_entries
    }
}

fn warp(matrix: CostMatrix) -> WarpedPath {
    // Shape validity is a `CostMatrix::new` invariant; fields are private.
    let (n, m) = (matrix.rows, matrix.cols);
    // cost[(i, j)] over a (n+1) x (m+1) grid with an infinite border,
    // Trace holds a closed move only for each completed interior cell.
    let width = m + 1;
    // f64 accumulation keeps finite f32 input costs finite along long paths.
    let mut cost = vec![f64::INFINITY; (n + 1) * width];
    let mut trace = Vec::with_capacity(n * m);
    cost[..width].fill(0.0);
    for i in 1..=n {
        for j in 1..=m {
            let c0 = cost[(i - 1) * width + (j - 1)];
            let c1 = cost[(i - 1) * width + j];
            let c2 = cost[i * width + (j - 1)];
            // Deterministic minimum: ties prefer diagonal, then text axis.
            let (best, mv) = if c0 <= c1 && c0 <= c2 {
                (c0, Step::Diagonal)
            } else if c1 <= c2 {
                (c1, Step::Text)
            } else {
                (c2, Step::Frame)
            };
            cost[i * width + j] = f64::from(matrix.at(i - 1, j - 1)) + best;
            trace.push(mv);
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
        match trace[(i - 1) * m + j - 1] {
            Step::Diagonal => {
                i -= 1;
                j -= 1;
            }
            Step::Text => i -= 1,
            Step::Frame => j -= 1,
        }
    }
    path.reverse();
    WarpedPath {
        matrix,
        points: path,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::expect_used)]
mod tests {
    use super::*;

    /// Guards: A cheap suffix may skip its audio prefix while retaining both labels and the fixed endpoint.
    /// Kind: policy
    /// Not a type: Private WarpedPath binds the completed traceback to its matrix, but cannot select the cheapest suffix or the intended open-begin endpoint rule from numerical costs.
    /// State: None. One admitted synthetic matrix exercises the pure alignment and consuming frame transfer.
    /// Concurrency: None. Matrix, traceback and result are owned by this call.
    #[test]
    fn open_begin_skips_prefix_but_covers_all_labels_and_fixed_end() {
        let matrix = CostMatrix::new(2, 4, vec![9., 9., -4., 9., 9., 9., 9., -4.]).unwrap();
        let path = warp(matrix);
        assert_eq!(
            path.points
                .last()
                .map(|point| (point.text_idx, point.time_idx)),
            Some((1, 3))
        );
        assert_eq!(path.into_alignment().into_row_entries(), vec![2, 3]);
    }

    /// Guards: The DP cost equals independently enumerated legal paths for every small signed matrix, with bounded monotone row entries.
    /// Kind: measurement
    /// Not a type: CostMatrix admits finite shape-consistent inputs and Step closes move alternatives, but the compiler cannot prove DP optimality or the numerical row-entry projection.
    /// State: None. Pure matrices cover every sign pattern through 3 by 3; one source-bound traceback supplies both the measured cost and consumed projection.
    /// Concurrency: None. Every matrix and completed path is local to one iteration.
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
                    let path = warp(matrix);
                    let observed: f64 = path
                        .points
                        .iter()
                        .map(|p| f64::from(path.matrix.at(p.text_idx, p.time_idx)))
                        .sum();
                    let expected = (0..cols)
                        .map(|col| cheapest(&path.matrix, 0, col))
                        .fold(f64::INFINITY, f64::min);
                    assert_eq!(observed, expected);
                    let entries = path.into_alignment();
                    assert_eq!(entries.row_entries().len(), rows);
                    assert!(entries.row_entries().windows(2).all(|p| p[0] <= p[1]));
                    assert!(entries.row_entries().iter().all(|&col| col < cols));
                }
            }
        }
    }

    /// Guards: Exact ties choose diagonal then text, while negative prefix costs may favor assigning the prefix.
    /// Kind: policy
    /// Not a type: The closed Step graph excludes unknown moves, but several legal paths have equal cost and require this explicit deterministic tie policy.
    /// State: None. Zero and negative synthetic costs exercise alternatives that an optimal-cost oracle cannot distinguish.
    /// Concurrency: None. Independent admitted matrices own all numerical state.
    #[test]
    fn repeated_labels_and_zero_cost_ties_do_not_drop_rows() {
        let matrix = CostMatrix::new(4, 2, vec![0.; 8]).unwrap();
        assert_eq!(
            LexicalFrameAlignment::align(matrix).row_entries(),
            &[0, 0, 0, 1]
        );
        // Negative costs may still favor assigning the prefix. Open-begin is
        // permission to skip, never evidence that a frame is physically silent.
        let matrix = CostMatrix::new(1, 4, vec![-1.; 4]).unwrap();
        assert_eq!(LexicalFrameAlignment::align(matrix).row_entries(), &[0]);
    }

    /// Guards: NaN, both infinities and overflowing declared dimensions are refused before alignment.
    /// Kind: boundary
    /// Not a type: The PyO3 adapter receives runtime NumPy dimensions and f32 bits. CostMatrix is the sole admission edge, whose refusal of those raw representations requires observation.
    /// State: None. Raw numerical buffers exercise admission without a Python process or model.
    /// Concurrency: None. Each malformed buffer and constructor result is local.
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

    /// Guards: A zero-length text axis produces the exact Empty shape diagnostic.
    /// Kind: boundary
    /// Not a type: A NumPy array can arrive with a zero dimension. Private admitted matrices prevent downstream use, while the raw-to-admitted constructor must still refuse the external shape accurately.
    /// State: None. An empty value buffer represents the shape admission case directly.
    /// Concurrency: None. The constructor reads no shared state.
    #[test]
    fn dtw_rejects_empty() {
        assert_eq!(
            CostMatrix::new(0, 3, vec![]).unwrap_err(),
            FaDtwError::Empty { rows: 0, cols: 3 }
        );
    }
}
