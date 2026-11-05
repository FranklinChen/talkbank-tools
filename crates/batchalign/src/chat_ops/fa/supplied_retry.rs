//! One terminal exact-source attempt, with exclusive ownership of missing slots.

use std::collections::BTreeMap;

use batchalign_transform::decisions::{DecisionRecord, DecisionStrategy, FaStrategy};
use talkbank_model::alignment::helpers::{
    TierDomain, WordItem, WordItemMut, counts_for_tier, walk_words, walk_words_mut,
};
use talkbank_model::model::{ChatFile, Utterance};

use super::injection::{TimingCursor, project_timing_for_word};
use super::orchestrate::FaImposed;
use super::{SuppliedRetrySource, WordTiming, get_word_timing, utterances_indexed_mut};

/// A terminal result. No result has a transition back to a request.
pub(crate) enum SuppliedRetryAnswer {
    Timings {
        timings: Vec<Option<WordTiming>>,
        source: String,
    },
    Unresolved {
        reason: String,
    },
}

struct MissingSlot {
    ordinal: usize,
    lower_ms: u64,
    upper_ms: u64,
}

struct MissingUtterance<'a> {
    line: usize,
    utterance: &'a mut Utterance,
    source: SuppliedRetrySource,
    slots: Vec<MissingSlot>,
}

/// Exclusive utterance borrows keep the admitted slot addresses stable across
/// async inference. Accepted slots grant no writing capability.
pub(crate) struct SuppliedRetrySession<'a, P> {
    imposed: FaImposed,
    pending: std::vec::IntoIter<(MissingUtterance<'a>, P)>,
}

pub(crate) enum SuppliedRetryStep<'a, P> {
    Complete(FaImposed),
    Pending(PendingSuppliedRetry<'a, P>),
}

pub(crate) struct PendingSuppliedRetry<'a, P> {
    preparation: P,
    current: MissingUtterance<'a>,
    session: SuppliedRetrySession<'a, P>,
}

impl<'a, P> SuppliedRetrySession<'a, P> {
    pub(crate) fn new(
        imposed: FaImposed,
        chat: &'a mut ChatFile,
        sources: impl Iterator<Item = (SuppliedRetrySource, P)>,
    ) -> Self {
        let mut sources: BTreeMap<_, _> = sources
            .map(|(source, preparation)| (source.index(), (source, preparation)))
            .collect();
        let mut pending = Vec::new();
        for (line, index, utterance) in utterances_indexed_mut(chat) {
            let Some((source, preparation)) = sources.remove(&index) else {
                continue;
            };
            let mut slots: Vec<MissingSlot> = Vec::new();
            let mut preceding_end = source.window().audio_start().get();
            let mut ordinal = 0;
            let mut open_start = 0;
            walk_words(&utterance.main.content.content, None, &mut |item| {
                let word = match item {
                    WordItem::Word(word) => word,
                    WordItem::ReplacedWord(replaced) => &replaced.word,
                    WordItem::Separator(_) => return,
                };
                match get_word_timing(word).filter(|span| span.end_ms > span.start_ms) {
                    Some(span) => {
                        // Every previous missing slot is bounded by this first
                        // retained successor, then removed from the open run.
                        for slot in &mut slots[open_start..] {
                            slot.upper_ms = span.start_ms;
                        }
                        open_start = slots.len();
                        preceding_end = preceding_end.max(span.end_ms);
                    }
                    None if counts_for_tier(word, TierDomain::Wor) => slots.push(MissingSlot {
                        ordinal,
                        lower_ms: preceding_end,
                        upper_ms: source.window().end().get(),
                    }),
                    None => {}
                }
                ordinal += 1;
            });
            if !slots.is_empty() {
                pending.push((
                    MissingUtterance {
                        line,
                        utterance,
                        source,
                        slots,
                    },
                    preparation,
                ));
            }
        }
        Self {
            imposed,
            pending: pending.into_iter(),
        }
    }

    pub(crate) fn next(mut self) -> SuppliedRetryStep<'a, P> {
        match self.pending.next() {
            Some((current, preparation)) => SuppliedRetryStep::Pending(PendingSuppliedRetry {
                current,
                preparation,
                session: self,
            }),
            None => SuppliedRetryStep::Complete(self.imposed),
        }
    }
}

impl<'a, P> PendingSuppliedRetry<'a, P> {
    /// Transfer preparation authority out exactly once. The awaiting phase
    /// cannot issue another preparation or return to this pending state.
    pub(crate) fn start(self) -> (AwaitingSuppliedRetry<'a, P>, P) {
        (
            AwaitingSuppliedRetry {
                current: self.current,
                session: self.session,
            },
            self.preparation,
        )
    }
}

pub(crate) struct AwaitingSuppliedRetry<'a, P> {
    current: MissingUtterance<'a>,
    session: SuppliedRetrySession<'a, P>,
}

impl<'a, P> AwaitingSuppliedRetry<'a, P> {
    pub(crate) fn source(&self) -> &SuppliedRetrySource {
        &self.current.source
    }

    /// Only missing slots can be written. A candidate must fit the source and
    /// retained neighbours without moving either. Every outcome is terminal.
    pub(crate) fn complete(mut self, answer: SuppliedRetryAnswer) -> SuppliedRetrySession<'a, P> {
        let target = self.current;
        let requested = target.slots.len();
        let mut filled = Vec::new();
        let detail = match answer {
            SuppliedRetryAnswer::Unresolved { reason } => reason,
            SuppliedRetryAnswer::Timings { timings, source }
                if timings.len() == target.source.words().len() =>
            {
                let mut slots = target.slots.into_iter().peekable();
                let mut cursor = TimingCursor::with_offset(&timings, 0);
                let mut ordinal = 0;
                let mut last_filled_end = target.source.window().audio_start().get();
                walk_words_mut(
                    target.utterance.main.content.content.as_mut_slice(),
                    None,
                    &mut |item| {
                        let word = match item {
                            WordItemMut::Word(word) => word,
                            WordItemMut::ReplacedWord(replaced) => &mut replaced.word,
                            WordItemMut::Separator(_) => return,
                        };
                        let projected = project_timing_for_word(word, &mut cursor);
                        if slots.peek().is_some_and(|slot| slot.ordinal == ordinal)
                            && let Some(slot) = slots.next()
                            && let Some(timing) = projected
                            && timing.start_ms >= slot.lower_ms.max(last_filled_end)
                            && timing.end_ms <= slot.upper_ms
                        {
                            word.inline_bullet = Some(timing.bullet());
                            last_filled_end = timing.end_ms;
                            filled.push(format!("word {ordinal}: {timing:?}"));
                        }
                        ordinal += 1;
                    },
                );
                source
            }
            SuppliedRetryAnswer::Timings { .. } => "refused mismatched reply population".into(),
        };
        let index = target.source.index();
        if !filled.is_empty() {
            self.session.imposed.retry_touched(index);
        }
        self.session.imposed.postprocess.push(DecisionRecord::new_and_trace(
            target.line,
            target.utterance.main.speaker.as_str().to_owned(),
            DecisionStrategy::Fa(FaStrategy::TimingProvenance),
            format!("terminal supplied-window retry u{} {}..{}ms: filled {}/{} missing AST slots; {}; {}",
                index.raw(), target.source.window().audio_start().get(), target.source.window().end().get(), filled.len(), requested, detail, filled.join("; ")),
            filled.len() != requested,
        ));
        self.session
    }
}

#[cfg(test)]
mod tests;
