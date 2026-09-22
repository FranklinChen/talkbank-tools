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
/// and whichever bounds it has.
///
/// A chunk with both bounds was projected with its timed neighbours; a chunk
/// missing either bound keeps what it had and goes downstream untimed on
/// that side, where the timing admission already handles a missing bound.
#[derive(Debug)]
pub(crate) struct MonotoneChunk<'a> {
    pub(crate) text: &'a str,
    pub(crate) start_s: Option<DurationSeconds>,
    pub(crate) end_s: Option<DurationSeconds>,
}

/// Chunk spans whose timed boundaries never decrease, built by
/// [`Self::project`] only.
#[derive(Debug)]
pub(crate) struct MonotoneChunkSpans<'a> {
    chunks: Vec<MonotoneChunk<'a>>,
    adjusted_boundaries: usize,
}

impl<'a> MonotoneChunkSpans<'a> {
    /// Project the fully timed producer spans onto the closest non-decreasing
    /// boundaries, in order; chunks without both bounds pass through as they
    /// are.
    pub(crate) fn project(raw: &'a [WhisperChunkSpanV2]) -> Self {
        let timed: Vec<(usize, f64, f64)> = raw
            .iter()
            .enumerate()
            .filter_map(|(index, chunk)| match (chunk.start_s, chunk.end_s) {
                (Some(start), Some(end)) => Some((index, start.get(), end.get())),
                _ => None,
            })
            .collect();
        let boundaries: Vec<f64> = timed
            .iter()
            .flat_map(|(_, start, end)| [*start, *end])
            .collect();
        let fitted = project_non_decreasing(&boundaries);
        let adjusted_boundaries = boundaries
            .iter()
            .zip(&fitted)
            .filter(|(before, after)| before != after)
            .count();
        let (pairs, _) = fitted.as_chunks::<2>();
        let mut projected = timed
            .iter()
            .zip(pairs)
            .map(|((index, _, _), [start_s, end_s])| (*index, *start_s, *end_s))
            .peekable();
        let chunks = raw
            .iter()
            .enumerate()
            .map(
                |(index, chunk)| match projected.next_if(|(timed, _, _)| *timed == index) {
                    Some((_, start_s, end_s)) => MonotoneChunk {
                        text: &chunk.text,
                        start_s: Some(DurationSeconds(start_s)),
                        end_s: Some(DurationSeconds(end_s)),
                    },
                    None => MonotoneChunk {
                        text: &chunk.text,
                        start_s: chunk.start_s.map(DurationSeconds::from),
                        end_s: chunk.end_s.map(DurationSeconds::from),
                    },
                },
            )
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
            start_s: Some(NonNegativeSeconds::try_from(start_s).expect("test bound")),
            end_s: Some(NonNegativeSeconds::try_from(end_s).expect("test bound")),
        }
    }

    fn untimed(text: &str) -> WhisperChunkSpanV2 {
        WhisperChunkSpanV2 {
            text: text.into(),
            start_s: None,
            end_s: None,
        }
    }

    fn spans_of(projected: &MonotoneChunkSpans) -> Vec<(f64, f64)> {
        projected
            .iter()
            .map(|c| {
                (
                    c.start_s.expect("timed in this test").0,
                    c.end_s.expect("timed in this test").0,
                )
            })
            .collect()
    }

    /// A chunk without both bounds is neither projected nor dropped: its
    /// words pass through untimed, and the timed chunks around it are
    /// projected as if it were not there.
    #[test]
    fn an_untimed_chunk_passes_through_and_does_not_disturb_the_projection() {
        let raw = [
            span("one", 0.0, 1.2),
            untimed("two"),
            span("three", 1.0, 2.0),
        ];
        let projected = MonotoneChunkSpans::project(&raw);
        let chunks: Vec<_> = projected.iter().collect();
        assert_eq!(chunks[1].text, "two");
        assert_eq!((chunks[1].start_s, chunks[1].end_s), (None, None));
        assert!((chunks[0].end_s.unwrap().0 - 1.1).abs() < 1e-9);
        assert!((chunks[2].start_s.unwrap().0 - 1.1).abs() < 1e-9);
        assert_eq!(projected.adjusted_boundaries(), 2);
    }

    /// The case that emptied Cantonese transcripts: one chunk, no timestamps
    /// at all. The words survive.
    #[test]
    fn a_whole_transcript_without_timestamps_is_kept_untimed() {
        let raw = [untimed("有個小朋友在戶外踢球")];
        let projected = MonotoneChunkSpans::project(&raw);
        let chunks: Vec<_> = projected.iter().collect();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "有個小朋友在戶外踢球");
        assert_eq!(projected.adjusted_boundaries(), 0);
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
