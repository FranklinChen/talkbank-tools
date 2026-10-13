//! Source-owned request admission for each piece of an anchored partition.

use super::{AnchoredPiece, FaWindow, FaWord, Ms, NonEmptyWords, RefusedWindow};
use crate::chat_ops::fa::utr::UtteranceAnchors;
use talkbank_model::WordIdx;

/// A consecutive part of the original word/window partition. Only the request
/// arm grants a dispatch capability; an unavailable arm still owns its words.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnchoredPart {
    /// Fits the engine budget and may be dispatched.
    Request(AnchoredPiece),
    /// No admitted request; retain these words untimed unless prior evidence
    /// legitimately supplies their timing.
    Unavailable(UnavailablePiece),
}

/// A source-bound word range whose anchor gap exceeds the engine budget.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnavailablePiece {
    words: NonEmptyWords,
    window: FaWindow,
    budget: Ms,
}

impl AnchoredPart {
    /// The partition producer alone can mint request capability. Classification
    /// consumes the words, so both outcomes retain exactly the same population.
    pub(super) fn admit(words: NonEmptyWords, window: FaWindow, budget: Ms) -> Self {
        if window.len() <= budget {
            Self::Request(AnchoredPiece { words, window })
        } else {
            Self::Unavailable(UnavailablePiece {
                words,
                window,
                budget,
            })
        }
    }

    /// Rescue only a fully anchored final word from an unavailable part.
    /// The preceding words must remain over budget and unrequested: this is
    /// not permission to crop a request for words whose timing is unknown.
    /// Existing requests and their cache identities are unchanged.
    pub(super) fn retain_anchored_tail(self, anchors: &UtteranceAnchors) -> (Option<Self>, Self) {
        let Self::Unavailable(piece) = self else {
            return (None, self);
        };
        let anchor = anchors.anchors().iter().find(|anchor| {
            anchor.word() == piece.words.last
                && anchor.end() == piece.window.end()
                && anchor.start() < anchor.end()
        });
        let Some(anchor) = anchor else {
            return (None, Self::Unavailable(piece));
        };
        let Some(previous) = piece.words.as_slice().iter().rev().nth(1) else {
            return (None, Self::Unavailable(piece));
        };
        let previous = previous.utterance_word_index;
        let Ok((prefix_window, tail_window)) = piece.window.split_at(anchor.start()) else {
            return (None, Self::Unavailable(piece));
        };
        if prefix_window.len() <= piece.budget || tail_window.len() > piece.budget {
            return (None, Self::Unavailable(piece));
        }
        let (prefix_words, tail_words) = match piece.words.split_after(previous) {
            Ok(parts) => parts,
            Err(words) => return (None, Self::Unavailable(UnavailablePiece { words, ..piece })),
        };
        (
            Some(Self::Unavailable(UnavailablePiece {
                words: prefix_words,
                window: prefix_window,
                budget: piece.budget,
            })),
            Self::Request(AnchoredPiece {
                words: tail_words,
                window: tail_window,
            }),
        )
    }

    /// All words of this part, whether or not it can be requested.
    pub fn words(&self) -> &[FaWord] {
        match self {
            Self::Request(piece) => piece.words(),
            Self::Unavailable(piece) => piece.words(),
        }
    }

    /// Its position in the original window partition, not dispatch permission.
    pub fn window(&self) -> FaWindow {
        match self {
            Self::Request(piece) => piece.window(),
            Self::Unavailable(piece) => piece.window(),
        }
    }

    /// First original word index, independent of request admission.
    pub fn first_word(&self) -> WordIdx {
        match self {
            Self::Request(piece) => piece.first_word(),
            Self::Unavailable(piece) => piece.first_word(),
        }
    }

    /// Last original word index, inclusive.
    pub fn last_word(&self) -> WordIdx {
        match self {
            Self::Request(piece) => piece.last_word(),
            Self::Unavailable(piece) => piece.last_word(),
        }
    }
}

impl UnavailablePiece {
    /// The original consecutive words, with original utterance word indices.
    pub fn words(&self) -> &[FaWord] {
        self.words.as_slice()
    }

    /// First word of the unavailable range (zero based).
    pub fn first_word(&self) -> WordIdx {
        self.words.first
    }

    /// Last word of the unavailable range (inclusive, zero based).
    pub fn last_word(&self) -> WordIdx {
        self.words.last
    }

    /// The unavailable span, inside the recording but over the request budget.
    pub fn window(&self) -> FaWindow {
        self.window
    }

    /// Durable reason for leaving this exact range unrequested.
    pub fn refusal(&self) -> RefusedWindow {
        RefusedWindow::AnchorGap {
            start_ms: self.window.audio_start().get(),
            end_ms: self.window.end().get(),
            budget_ms: self.budget.0,
            gap_start_ms: self.window.audio_start().get(),
            gap_end_ms: self.window.end().get(),
        }
    }
}
