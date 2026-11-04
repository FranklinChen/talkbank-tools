//! Reconcile independent observations only where their source audio overlaps.
//!
//! Each Whisper window is decoded on its own and shares some audio with its
//! neighbours. A word both windows heard, at the same time, in the shared
//! audio, is one word; the reconciler keeps it once. When it cannot tell
//! WHICH earlier observation an incoming one repeats (two fit, or the one
//! that fits is already claimed, or the matches cross), it keeps the earlier
//! window's copy, drops the later window's, and records the decision as a
//! [`SeamAmbiguity`] for the transcript and the job to carry. Until
//! 2026-11-03 that case refused the whole recording.

use std::borrow::Cow;

use batchalign_transform::asr_postprocess::{SeamAmbiguity, WindowOrdinal};
use batchalign_types::api::AudioPositionSeconds;
use batchalign_types::interval::AdmittedInterval;
use batchalign_types::worker_v2::{WhisperChunkSpanV2, WhisperObservationsV2};

/// A window's offset put one of its observations past any audio position.
#[derive(Debug, thiserror::Error)]
#[error("Whisper window offset does not fit an audio position")]
pub(super) struct WindowMergeRefusal;

/// The settled observations of a recording, and every seam decision the
/// settling made by rule rather than by evidence.
pub(super) struct Reconciled<'a> {
    /// The observations in recording order, each shared word once.
    pub(super) spans: Cow<'a, [WhisperChunkSpanV2]>,
    /// Each repetition at a seam that was kept once, in recording order.
    pub(super) seams: Vec<SeamAmbiguity>,
}

struct Observation {
    span: WhisperChunkSpanV2,
    window_start: f64,
    window_end: f64,
}

enum SeamObservation {
    Shared(usize),
    Novel(Observation),
}

impl Observation {
    /// Whether `other` repeats this observation: same text, both timed, both
    /// inside the audio the two windows share, and intersecting. Answers with
    /// THIS observation's interval, admitted, so a caller that keeps this copy
    /// has its place without a second look at optional fields. An interval
    /// the admission refuses (a position beyond its range) is no authority to
    /// remove a word either, so it answers as no duplicate.
    fn duplicated_by(&self, other: &Self) -> Option<AdmittedInterval> {
        if self.span.text.trim() != other.span.text.trim() || self.span.text.trim().is_empty() {
            return None;
        }
        let (Some(a), Some(b), Some(c), Some(d)) = (
            self.span.start_s,
            self.span.end_s,
            other.span.start_s,
            other.span.end_s,
        ) else {
            // Missing timing provides no authority to remove a word.
            return None;
        };
        // Strict intersection of BOTH observations and BOTH physical windows.
        // Forward timestamps alone are not evidence of a duplicate utterance.
        let shared_start = self.window_start.max(other.window_start);
        let shared_end = self.window_end.min(other.window_end);
        let duplicate = a.get() >= shared_start
            && c.get() >= shared_start
            && b.get() <= shared_end
            && d.get() <= shared_end
            && a.get().max(c.get()) < b.get().min(d.get());
        if !duplicate {
            return None;
        }
        AdmittedInterval::admit_positions(a, b).ok()
    }
}

/// Preserve each input sequence, using time only to interleave unmatched words.
fn interleave(left: Vec<Observation>, right: Vec<Observation>) -> Vec<Observation> {
    let mut left = left.into_iter().peekable();
    let mut right = right.into_iter().peekable();
    let mut next = Vec::new();
    while let (Some(a), Some(b)) = (left.peek(), right.peek()) {
        let right_first = match (a.span.start_s, b.span.start_s) {
            (Some(a), Some(b)) => b.get() < a.get(),
            _ => false,
        };
        if right_first {
            next.extend(right.next());
        } else {
            next.extend(left.next());
        }
    }
    next.extend(left);
    next.extend(right);
    next
}

pub(super) fn reconcile(raw: &WhisperObservationsV2) -> Result<Reconciled<'_>, WindowMergeRefusal> {
    let windows = match raw {
        WhisperObservationsV2::Monolithic(spans) => {
            return Ok(Reconciled {
                spans: Cow::Borrowed(spans),
                seams: Vec::new(),
            });
        }
        WhisperObservationsV2::Windows(windows) => windows,
    };
    let rate = f64::from(windows.sample_rate_hz().0);
    let mut merged: Vec<Observation> = Vec::new();
    let mut seams = Vec::new();
    for (ordinal, window) in windows.windows().iter().enumerate() {
        let start = window.start_sample.0 as f64 / rate;
        let end = start + window.sample_count.0 as f64 / rate;
        // The seam: the first instant this window owns, which is where the
        // previous window's own stretch of the recording ends.
        let seam_ms =
            ((window.start_sample.0 + window.left_context.0) as f64 / rate * 1000.0).round() as i64;
        let shift = |bound: Option<AudioPositionSeconds>| {
            bound
                .map(|bound| {
                    AudioPositionSeconds::try_from(start + bound.get())
                        .map_err(|_| WindowMergeRefusal)
                })
                .transpose()
        };
        let incoming = window
            .spans
            .iter()
            .map(|span| {
                Ok(Observation {
                    span: WhisperChunkSpanV2 {
                        text: span.text.clone(),
                        start_s: shift(span.start_s)?,
                        end_s: shift(span.end_s)?,
                    },
                    window_start: start,
                    window_end: end,
                })
            })
            .collect::<Result<Vec<_>, WindowMergeRefusal>>()?;

        let mut used = vec![false; merged.len()];
        let mut incoming_plan = Vec::new();
        let mut previous_match = None;
        for observation in incoming {
            let mut matches = merged.iter().enumerate().filter_map(|(index, old)| {
                old.duplicated_by(&observation).map(|kept| (index, kept))
            });
            let Some((index, kept)) = matches.next() else {
                incoming_plan.push(SeamObservation::Novel(observation));
                continue;
            };
            // Two earlier observations fit this one, or the one that fits is
            // already claimed by an earlier incoming word, or the matches
            // cross: which observation is the repeat cannot be told from the
            // evidence. The earlier window's copy stays, this one is dropped,
            // and the decision is recorded for the transcript and the job.
            let ambiguous = matches.next().is_some()
                || used[index]
                || previous_match.is_some_and(|previous| index <= previous);
            if ambiguous {
                seams.push(SeamAmbiguity::new(
                    seam_ms,
                    &observation.span.text,
                    WindowOrdinal::new(ordinal),
                    kept,
                ));
                continue;
            }
            used[index] = true;
            previous_match = Some(index);
            incoming_plan.push(SeamObservation::Shared(index));
        }
        // Shared observations are ordered anchors. In particular an untimed
        // word between two anchors must stay between them, not drift to the
        // end just because it cannot participate in a timestamp comparison.
        let mut left = merged.into_iter().enumerate().peekable();
        let mut pending = Vec::new();
        let mut next = Vec::new();
        for item in incoming_plan {
            match item {
                SeamObservation::Novel(observation) => pending.push(observation),
                SeamObservation::Shared(anchor) => {
                    let mut prefix = Vec::new();
                    while let Some((_, observation)) = left.next_if(|(index, _)| *index < anchor) {
                        prefix.push(observation);
                    }
                    next.extend(interleave(prefix, std::mem::take(&mut pending)));
                    next.extend(left.next().map(|(_, observation)| observation));
                }
            }
        }
        next.extend(interleave(
            left.map(|(_, observation)| observation).collect(),
            pending,
        ));
        merged = next;
    }
    Ok(Reconciled {
        spans: Cow::Owned(merged.into_iter().map(|item| item.span).collect()),
        seams,
    })
}
