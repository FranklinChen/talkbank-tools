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
            Self::Request(AnchoredPiece::new(words, window))
        } else {
            Self::Unavailable(UnavailablePiece {
                words,
                window,
                budget,
            })
        }
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
            Self::Request(piece) => piece.ownership_window(),
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
    /// A recovered final word may join the FOLLOWING supported request. It
    /// never becomes an isolated request over just its ASR token interval.
    /// The unknown prefix remains unavailable, with its original words.
    pub(super) fn recover_into(
        self,
        next: AnchoredPiece,
        anchors: &UtteranceAnchors,
    ) -> (Self, AnchoredPiece) {
        if self.window.end() != next.ownership_window.audio_start()
            || self.words.last.raw() + 1 != next.first_word().raw()
        {
            return (self, next);
        }
        let Some(anchor) = anchors.anchors().iter().find(|anchor| {
            anchor.word() == self.words.last
                && anchor.end() == self.window.end()
                && anchor.start() < anchor.end()
        }) else {
            return (self, next);
        };
        let Some(previous) = self.words.as_slice().iter().rev().nth(1) else {
            return (self, next);
        };
        let previous = previous.utterance_word_index;
        let Ok((prefix_window, _)) = self.window.split_at(anchor.start()) else {
            return (self, next);
        };
        let Ok(joined_window) =
            FaWindow::within(&self.window.recording(), anchor.start(), next.window.end())
        else {
            return (self, next);
        };
        if prefix_window.len() <= self.budget || joined_window.len() > self.budget {
            return (self, next);
        }
        let (prefix, mut tail) = match self.words.split_after(previous) {
            Ok(parts) => parts,
            Err(words) => return (Self { words, ..self }, next),
        };
        tail.last = next.words.last;
        tail.words.extend(next.words.into_vec());
        (
            Self {
                words: prefix,
                window: prefix_window,
                budget: self.budget,
            },
            AnchoredPiece::new(tail, joined_window),
        )
    }

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
