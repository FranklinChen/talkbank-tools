//! Whisper chunk spans with non-decreasing boundaries.
//!
//! Two producers hand the pipeline `WhisperChunkSpanV2` lists: the Python
//! HuggingFace worker and the in-process whisper.cpp backend. Both stitch audio
//! chunks, and at a seam a word can start before the previous word ends; a
//! chunk can even arrive with its end before its start. Neither producer is
//! asked to repair that. The repair belongs where the spans are CONSUMED:
//! [`MonotoneChunkSpans::project`] is the only input
//! [`crate::worker::asr_result_v2::whisper_chunk_result_to_asr_response`]
//! lowers, so a list that has not been through it has no type the lowering
//! accepts, whichever producer made it.
//!
//! The projection is L2 isotonic regression over the flat boundary sequence
//! `start_0, end_0, start_1, end_1, ...` (pool-adjacent violators). It moves
//! only the boundaries that conflict and never reorders chunks. An inverted
//! chunk pools its two bounds, so it comes out zero-width at their midpoint;
//! the timing admission downstream (`WordTiming::from_admitted`) then demotes
//! a zero-width span to untimed, which is the right end for a chunk whose
//! producer could not say when it was. `tests::projection_matches_the_exact_oracle`
//! checks the fit against an independent closed form on every short sequence.

use crate::api::DurationSeconds;
use crate::types::worker_v2::WhisperChunkSpanV2;

/// L2 isotonic regression by pool-adjacent violators, O(n).
///
/// Blocks carry `(sum, count)` so a merge is an addition; means are taken
/// once, when the blocks are expanded.
pub(crate) fn project_non_decreasing(values: &[f64]) -> Vec<f64> {
    let mut blocks: Vec<(f64, usize)> = Vec::with_capacity(values.len());
    for &value in values {
        let mut block = (value, 1);
        // Pool with the block before while that block's mean is larger.
        while let Some(&(sum, count)) = blocks.last()
            && sum / count as f64 > block.0 / block.1 as f64
        {
            blocks.pop();
            block = (block.0 + sum, block.1 + count);
        }
        blocks.push(block);
    }
    blocks
        .into_iter()
        .flat_map(|(sum, count)| std::iter::repeat_n(sum / count as f64, count))
        .collect()
}

/// One chunk after projection: its text, borrowed from the producer's span,
/// and its settled bounds.
#[derive(Debug)]
pub(crate) struct MonotoneChunk<'a> {
    pub(crate) text: &'a str,
    pub(crate) start_s: DurationSeconds,
    pub(crate) end_s: DurationSeconds,
}

/// Chunk spans whose boundaries never decrease, built by [`Self::project`] only.
#[derive(Debug)]
pub(crate) struct MonotoneChunkSpans<'a> {
    chunks: Vec<MonotoneChunk<'a>>,
    adjusted_boundaries: usize,
}

impl<'a> MonotoneChunkSpans<'a> {
    /// Project raw producer spans onto the closest non-decreasing boundaries.
    pub(crate) fn project(raw: &'a [WhisperChunkSpanV2]) -> Self {
        let boundaries: Vec<f64> = raw
            .iter()
            .flat_map(|chunk| [chunk.start_s.get(), chunk.end_s.get()])
            .collect();
        let fitted = project_non_decreasing(&boundaries);
        let adjusted_boundaries = boundaries
            .iter()
            .zip(&fitted)
            .filter(|(before, after)| before != after)
            .count();
        let (pairs, _) = fitted.as_chunks::<2>();
        let chunks = raw
            .iter()
            .zip(pairs)
            .map(|(chunk, [start_s, end_s])| MonotoneChunk {
                text: &chunk.text,
                start_s: DurationSeconds(*start_s),
                end_s: DurationSeconds(*end_s),
            })
            .collect();
        Self {
            chunks,
            adjusted_boundaries,
        }
    }

    /// How many boundaries the projection moved; zero when the input was
    /// already monotone.
    pub(crate) fn adjusted_boundaries(&self) -> usize {
        self.adjusted_boundaries
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = &MonotoneChunk<'a>> {
        self.chunks.iter()
    }
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::api::NonNegativeSeconds;

    /// Exact L2 isotonic fit, independent of the algorithm under test:
    /// `fit[i] = max over j <= i of min over k >= i of mean(x[j..=k])`.
    fn oracle(values: &[f64]) -> Vec<f64> {
        (0..values.len())
            .map(|i| {
                (0..=i)
                    .map(|j| {
                        (i..values.len())
                            .map(|k| values[j..=k].iter().sum::<f64>() / (k - j + 1) as f64)
                            .fold(f64::INFINITY, f64::min)
                    })
                    .fold(f64::NEG_INFINITY, f64::max)
            })
            .collect()
    }

    fn span(text: &str, start_s: f64, end_s: f64) -> WhisperChunkSpanV2 {
        WhisperChunkSpanV2 {
            text: text.into(),
            start_s: NonNegativeSeconds::try_from(start_s).expect("test bound"),
            end_s: NonNegativeSeconds::try_from(end_s).expect("test bound"),
        }
    }

    fn spans_of(projected: &MonotoneChunkSpans) -> Vec<(f64, f64)> {
        projected.iter().map(|c| (c.start_s.0, c.end_s.0)).collect()
    }

    #[test]
    fn projection_matches_the_exact_oracle_on_every_short_sequence() {
        const VALUES: [f64; 4] = [0.0, 1.0, 2.0, 3.0];
        for length in 1..=6u32 {
            // Every sequence of `length` values over VALUES, read off one
            // base-4 code as its digits.
            for code in 0..VALUES.len().pow(length) {
                let values: Vec<f64> = (0..length)
                    .map(|digit| VALUES[(code / VALUES.len().pow(digit)) % VALUES.len()])
                    .collect();
                let got = project_non_decreasing(&values);
                let want = oracle(&values);
                assert!(
                    got.iter().zip(&want).all(|(g, w)| (g - w).abs() < 1e-9),
                    "{values:?}: got {got:?}, want {want:?}"
                );
                assert!(got.windows(2).all(|w| w[0] <= w[1] + 1e-9), "{values:?}");
            }
        }
    }

    #[test]
    fn overlapping_chunks_are_reconciled_at_the_seam_only() {
        let raw = [
            span("one", 0.0, 1.2),
            span("two", 1.0, 2.0),
            span("three", 2.0, 3.0),
        ];
        let projected = MonotoneChunkSpans::project(&raw);
        let spans = spans_of(&projected);
        assert_eq!(spans[0].0, 0.0);
        assert_eq!(spans[2], (2.0, 3.0));
        assert!((spans[0].1 - 1.1).abs() < 1e-9 && (spans[1].0 - 1.1).abs() < 1e-9);
        assert!(spans.windows(2).all(|w| w[0].1 <= w[1].0));
        assert_eq!(projected.adjusted_boundaries(), 2);
    }

    /// An inverted chunk pools its two bounds and comes out zero-width. That
    /// intermediate value is not what the pipeline keeps: the timing admission
    /// demotes a zero-width span to untimed (`ZeroLengthSpan`), so the word is
    /// placed by its neighbours rather than by a guess at which bound was
    /// right.
    #[test]
    fn an_inverted_chunk_collapses_to_zero_width_for_the_admission_to_demote() {
        let raw = [span("x", 2.0, 1.0)];
        let projected = MonotoneChunkSpans::project(&raw);
        assert_eq!(spans_of(&projected), vec![(1.5, 1.5)]);
        assert_eq!(projected.adjusted_boundaries(), 2);
    }

    #[test]
    fn a_monotone_input_is_untouched_and_reports_nothing_moved() {
        let raw = [span("a", 0.0, 0.5), span("b", 0.5, 1.25)];
        let projected = MonotoneChunkSpans::project(&raw);
        assert_eq!(spans_of(&projected), vec![(0.0, 0.5), (0.5, 1.25)]);
        assert_eq!(projected.adjusted_boundaries(), 0);
        assert_eq!(
            projected.iter().map(|c| c.text).collect::<Vec<_>>(),
            vec!["a", "b"]
        );
    }
}
