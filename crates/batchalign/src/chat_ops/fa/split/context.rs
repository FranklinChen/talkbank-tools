//! Borrow acoustic AND lexical context without changing word ownership.
//!
//! Context maximizes available duration within the existing budget, using
//! source-bound anchor starts/ends or the supported run's edges. It never
//! crosses an unavailable range. This is a search policy, not an assertion
//! that ASR token boundaries are exact acoustic boundaries.

use super::{AnchoredPart, AnchoredPiece, FaWindow, Ms, NonEmptyWords};
use crate::chat_ops::fa::coordinates::FileMs;
use crate::chat_ops::fa::utr::UtteranceAnchors;
use talkbank_model::WordIdx;

#[derive(Clone, Copy)]
struct Edge {
    word: WordIdx,
    at: FileMs,
}

/// Both word and time boundaries, admitted together by the context planner.
struct Selection {
    first: Edge,
    last: Edge,
}

pub(super) fn plan(
    parts: Vec<AnchoredPart>,
    anchors: &UtteranceAnchors,
    budget: Ms,
) -> Vec<AnchoredPart> {
    // Tail recovery happens before context borrowing. The recovered word and
    // following words become one owned piece, not duplicate acoustic calls.
    let mut supported = Vec::with_capacity(parts.len());
    for part in parts {
        match (supported.pop(), part) {
            (Some(AnchoredPart::Unavailable(before)), AnchoredPart::Request(after)) => {
                let (before, after) = before.recover_into(after, anchors);
                supported.push(AnchoredPart::Unavailable(before));
                supported.push(AnchoredPart::Request(after));
            }
            (Some(previous), part) => {
                supported.push(previous);
                supported.push(part);
            }
            (None, part) => supported.push(part),
        }
    }
    let mut remaining = supported.as_mut_slice();
    while !remaining.is_empty() {
        let count = remaining
            .iter()
            .take_while(|part| matches!(part, AnchoredPart::Request(_)))
            .count();
        if count == 0 {
            remaining = &mut remaining[1..];
            continue;
        }
        let (run, rest) = remaining.split_at_mut(count);
        expand_run(run, anchors, budget);
        remaining = rest;
    }
    supported
}

fn expand_run(parts: &mut [AnchoredPart], anchors: &UtteranceAnchors, budget: Ms) {
    let (Some(first), Some(last)) = (parts.first(), parts.last()) else {
        return;
    };
    let first = Edge {
        word: first.first_word(),
        at: first.window().audio_start(),
    };
    let last = Edge {
        word: last.last_word(),
        at: last.window().end(),
    };
    let words: Vec<_> = parts
        .iter()
        .flat_map(AnchoredPart::words)
        .cloned()
        .collect();
    let mut starts = vec![first];
    let mut ends = Vec::new();
    for anchor in anchors.anchors() {
        if anchor.word() < first.word
            || anchor.word() > last.word
            || anchor.start() >= anchor.end()
            || anchor.start() < first.at
            || anchor.end() > last.at
        {
            continue;
        }
        starts.push(Edge {
            word: anchor.word(),
            at: anchor.start(),
        });
        ends.push(Edge {
            word: anchor.word(),
            at: anchor.end(),
        });
    }
    ends.push(last);
    for part in parts {
        let AnchoredPart::Request(piece) = part else {
            continue;
        };
        let mut selected = Selection {
            first: Edge {
                word: piece.first_word(),
                at: piece.ownership_window.audio_start(),
            },
            last: Edge {
                word: piece.last_word(),
                at: piece.ownership_window.end(),
            },
        };
        let mut longest = piece.window.len();
        for start in &starts {
            if start.word > piece.first_word() || start.at > piece.ownership_window.audio_start() {
                break;
            }
            let limit = start.at.get().saturating_add(budget.0);
            let end_count = ends.partition_point(|end| end.at.get() <= limit);
            let Some(end) = end_count.checked_sub(1).and_then(|index| ends.get(index)) else {
                continue;
            };
            if end.word < piece.last_word() || end.at < piece.ownership_window.end() {
                continue;
            }
            let length = end.at.since(start.at);
            if length > longest {
                selected = Selection {
                    first: *start,
                    last: *end,
                };
                longest = length;
            }
        }
        apply(piece, &words, first.word, selected);
    }
}

/// The only context-installation transition. Its lexical slice and owned
/// subrange come from the same original supported population; no caller can
/// supply an unrelated ownership range beside model words.
fn apply(
    piece: &mut AnchoredPiece,
    run: &[super::FaWord],
    run_first: WordIdx,
    selected: Selection,
) {
    let first = selected.first.word.raw() - run_first.raw();
    let last = selected.last.word.raw() - run_first.raw();
    let Some(context) = run.get(first..=last) else {
        return;
    };
    let owned_start = piece.first_word().raw() - selected.first.word.raw();
    let owned = owned_start..owned_start + piece.words().len();
    // Boundary validation, not a second word-matching algorithm. If an
    // internal caller ever changes the population, retain the original piece.
    if context.get(owned.clone()) != Some(piece.words()) {
        return;
    }
    let Ok(window) = FaWindow::within(
        &piece.window.recording(),
        selected.first.at,
        selected.last.at,
    ) else {
        return;
    };
    let Some(words) = NonEmptyWords::from_vec(context.to_vec()) else {
        return;
    };
    piece.words = words;
    piece.owned = owned;
    piece.window = window;
}
