//! Reconcile independent observations only where their source audio overlaps.

use std::borrow::Cow;

use batchalign_types::api::AudioPositionSeconds;
use batchalign_types::worker_v2::{WhisperChunkSpanV2, WhisperObservationsV2};

#[derive(Debug, thiserror::Error)]
pub(super) enum WindowMergeRefusal {
    #[error("Whisper window offset does not fit an audio position")]
    Offset,
    #[error("Whisper overlap has ambiguous repeated observations")]
    Ambiguous,
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
    fn duplicates(&self, other: &Self) -> bool {
        if self.span.text.trim() != other.span.text.trim() || self.span.text.trim().is_empty() {
            return false;
        }
        let (Some(a), Some(b), Some(c), Some(d)) = (
            self.span.start_s,
            self.span.end_s,
            other.span.start_s,
            other.span.end_s,
        ) else {
            // Missing timing provides no authority to remove a word.
            return false;
        };
        // Strict intersection of BOTH observations and BOTH physical windows.
        // Forward timestamps alone are not evidence of a duplicate utterance.
        let shared_start = self.window_start.max(other.window_start);
        let shared_end = self.window_end.min(other.window_end);
        a.get() >= shared_start
            && c.get() >= shared_start
            && b.get() <= shared_end
            && d.get() <= shared_end
            && a.get().max(c.get()) < b.get().min(d.get())
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

pub(super) fn reconcile(
    raw: &WhisperObservationsV2,
) -> Result<Cow<'_, [WhisperChunkSpanV2]>, WindowMergeRefusal> {
    let windows = match raw {
        WhisperObservationsV2::Monolithic(spans) => return Ok(Cow::Borrowed(spans)),
        WhisperObservationsV2::Windows(windows) => windows,
    };
    let rate = f64::from(windows.sample_rate_hz().0);
    let mut merged: Vec<Observation> = Vec::new();
    for window in windows.windows() {
        let start = window.start_sample.0 as f64 / rate;
        let end = start + window.sample_count.0 as f64 / rate;
        let shift = |bound: Option<AudioPositionSeconds>| {
            bound
                .map(|bound| {
                    AudioPositionSeconds::try_from(start + bound.get())
                        .map_err(|_| WindowMergeRefusal::Offset)
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
            let mut matches = merged
                .iter()
                .enumerate()
                .filter(|(_, old)| old.duplicates(&observation))
                .map(|(index, _)| index);
            if let Some(index) = matches.next() {
                if matches.next().is_some()
                    || used[index]
                    || previous_match.is_some_and(|previous| index <= previous)
                {
                    return Err(WindowMergeRefusal::Ambiguous);
                }
                used[index] = true;
                previous_match = Some(index);
                incoming_plan.push(SeamObservation::Shared(index));
            } else {
                incoming_plan.push(SeamObservation::Novel(observation));
            }
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
    Ok(Cow::Owned(
        merged.into_iter().map(|item| item.span).collect(),
    ))
}
