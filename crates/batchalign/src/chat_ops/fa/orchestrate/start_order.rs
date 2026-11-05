//! Source-bound start-order repair. Immutable facts can be shared; a kept
//! source has no mutation target, and revisable targets are consumed once.

use super::super::{overlaps_previous_utterance, utterances_indexed_mut};
use super::*;
use std::sync::Arc;

struct StartOrderEntry {
    line_idx: usize,
    utterance_idx: UtteranceIdx,
    speaker: String,
    start_ms: u64,
    end_ms: u64,
    /// What the transcriber said about this utterance's relation to the one
    /// before it: the evidence a start regression is resolved by.
    overlap: TranscribedOverlap,
}

/// Immutable start facts stay paired with this exclusive utterance borrow.
/// Cloning the facts never clones permission to mutate the source.
struct StartOrderTarget<'utterance> {
    facts: Arc<StartOrderEntry>,
    utterance: &'utterance mut Utterance,
}

/// A kept start has no mutation capability. Only a revisable source can
/// produce a target accepted by start-regression classification.
enum StartOrderSource<'utterance> {
    Kept(Arc<StartOrderEntry>),
    Revisable(StartOrderTarget<'utterance>),
}
impl StartOrderSource<'_> {
    fn facts(&self) -> &Arc<StartOrderEntry> {
        match self {
            Self::Kept(facts) => facts,
            Self::Revisable(target) => &target.facts,
        }
    }
}

enum StartOrderPosition<'utterance> {
    Accepted(Arc<StartOrderEntry>),
    Regressing(StartRegression<'utterance>),
    Yielding(StartYield<'utterance>),
}

/// Only classification can bind a revisable target to a conflicting kept
/// boundary. Consuming resolution cannot repeat or mutate a kept utterance.
struct StartYield<'utterance> {
    target: StartOrderTarget<'utterance>,
    kept: Arc<StartOrderEntry>,
}
struct ResolvedStartYield {
    kept: Arc<StartOrderEntry>,
    timing: StrippedTiming,
}
impl StartYield<'_> {
    fn resolve(self) -> ResolvedStartYield {
        ResolvedStartYield {
            kept: self.kept,
            timing: strip_utterance_timing(self.target.utterance),
        }
    }
}

impl<'utterance> StartOrderTarget<'utterance> {
    fn classify(
        self,
        previous: Option<Arc<StartOrderEntry>>,
        next_kept: Option<Arc<StartOrderEntry>>,
    ) -> StartOrderPosition<'utterance> {
        if let Some(previous) = previous
            && self.facts.start_ms < previous.start_ms
        {
            return StartOrderPosition::Regressing(StartRegression {
                entry: self,
                previous,
            });
        }
        if let Some(kept) = next_kept
            && self.facts.start_ms > kept.start_ms
        {
            return StartOrderPosition::Yielding(StartYield { target: self, kept });
        }
        // Accepted source returns immutable facts only, never another target
        // that could be classified or stripped again.
        StartOrderPosition::Accepted(self.facts)
    }
}

/// Whether the transcriber marked an utterance as overlapping the utterance
/// before it. Read from the parsed model's linkers, where Chatter puts the
/// `+<` ("lazy overlap precedes") marker; never a flag a caller sets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TranscribedOverlap {
    /// The main tier begins with `+<`: this turn starts while the previous
    /// turn is still going.
    PrecedesMarked,
    /// No such marker: the transcriber placed this turn after the previous.
    Unmarked,
}

impl TranscribedOverlap {
    fn of(utterance: &Utterance) -> Self {
        match overlaps_previous_utterance(utterance) {
            true => Self::PrecedesMarked,
            false => Self::Unmarked,
        }
    }
}

/// Why a start regression was resolved by stripping rather than raising:
/// named where the choice is made, so the decision record cannot re-derive
/// it from other facts and drift.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StripCause {
    /// The utterance carries no `+<` marker: the transcriber placed it after
    /// the previous turn, so an earlier measured start is more likely a
    /// misplaced request than real speech.
    NoOverlapMarker,
    /// The utterance is marked `+<`, but nothing of it lies after the
    /// preceding start: the bullet ends at or before it, or every timed word
    /// does, so a raise would keep a bullet with no timed word inside it.
    NothingAfterPrecedingStart,
}

impl StripCause {
    /// The `cause=` value in the decision reason.
    fn as_str(self) -> &'static str {
        match self {
            Self::NoOverlapMarker => "no_overlap_marker",
            Self::NothingAfterPrecedingStart => "nothing_after_preceding_start",
        }
    }
}

/// A revisable bullet whose measured start precedes the greatest start
/// accepted before it. CHAT's start-order rule forbids writing that start on
/// this line, so the regression must be resolved before the file is written.
/// Minted only by [`StartOrderTarget::classify`], which owns the comparison, so holding one
/// proves the regression; [`Self::resolve`] is the only way out, and which
/// way it goes is decided by the utterance's own overlap evidence, not by
/// the caller.
struct StartRegression<'utterance> {
    /// The regressing utterance and its sole mutation capability.
    entry: StartOrderTarget<'utterance>,
    /// Immutable facts for the greatest preceding start, not mutation authority.
    previous: Arc<StartOrderEntry>,
}

/// How a [`StartRegression`] was resolved in the file.
#[derive(Debug)]
enum StartRegressionResolution {
    /// The whole utterance's timing was removed, for the named cause.
    Stripped {
        /// What the strip removed.
        timing: StrippedTiming,
        /// Why a raise was not made.
        cause: StripCause,
    },
    /// The transcriber marked the overlap (`+<`): the bullet now starts at
    /// the preceding start, words wholly before it lost their timing, the
    /// word straddling it was cut to begin there, and the rest keep their
    /// measured times.
    RaisedToPrecedingStart {
        /// Words whose measured interval ended at or before the new start.
        words_dropped: Vec<DroppedWordTiming>,
        /// Words cut to begin at the new start, with what they lost.
        words_trimmed: Vec<TrimmedWordTiming>,
    },
}

impl<'utterance> StartRegression<'utterance> {
    /// Mutate the actual borrowed utterance. A line-index lookup cannot
    /// substitute another document or manufacture an empty removal record.
    fn resolve(self) -> StartRegressionResolution {
        let stripped = |timing, cause| StartRegressionResolution::Stripped { timing, cause };
        let utterance = self.entry.utterance;
        let new_start = self.previous.start_ms;
        match self.entry.facts.overlap {
            TranscribedOverlap::Unmarked => stripped(
                strip_utterance_timing(utterance),
                StripCause::NoOverlapMarker,
            ),
            TranscribedOverlap::PrecedesMarked => {
                // A raise keeps the bullet only if it still has extent past
                // the new start and, when any word is timed, at least one
                // timed word ends after it. An utterance with a bullet and no
                // timed word at all (a transcript carrying bullets alone) is
                // raised: the raise loses nothing it has.
                let timed_extent_survives = furthest_word_timing_end(utterance)
                    .is_none_or(|furthest_end| furthest_end > new_start);
                if self.entry.facts.end_ms <= new_start || !timed_extent_survives {
                    return stripped(
                        strip_utterance_timing(utterance),
                        StripCause::NothingAfterPrecedingStart,
                    );
                }
                let cut = raise_bullet_start(utterance, new_start);
                StartRegressionResolution::RaisedToPrecedingStart {
                    words_dropped: cut.dropped,
                    words_trimmed: cut.trimmed,
                }
            }
        }
    }
}

/// Pass 1: file-order start monotonicity.
///
/// Kept starts are a FIXED skeleton. A revisable bullet must start at or
/// after the greatest start accepted before it and at or before the next kept
/// start after it; one that falls outside is the one resolved, recorded as a
/// start regression (below the preceding start; see [`StartRegression`],
/// which raises an utterance the transcriber marked `+<` and strips any
/// other) or as having yielded to the next kept start (above it). A kept
/// start that regresses below an earlier
/// kept start cannot be fixed without changing a kept bullet, so it is left
/// and recorded as unresolved; it does not lower the running start, so what
/// follows is judged as though it had been stripped. With no kept bullet this
/// is exactly the established rule: strip every start below the greatest
/// preceding accepted start.
pub(super) fn enforce_start_order(
    chat_file: &mut ChatFile,
    imposed: ImposedBullets<'_>,
    decisions: &mut Vec<batchalign_transform::decisions::DecisionRecord>,
    effects: &mut Vec<MonotonicityEffect>,
) -> Result<(), KeptBulletError> {
    use batchalign_transform::decisions::MonotonicityStrategy;

    let mut entries = Vec::new();
    for (line_idx, utterance_idx, utterance) in utterances_indexed_mut(chat_file) {
        let Some(bullet) = utterance
            .main
            .content
            .bullet
            .as_ref()
            .and_then(talkbank_model::BulletState::as_parsed)
        else {
            continue;
        };
        let mutability = imposed.mutability(utterance_idx, bullet)?;
        let facts = Arc::new(StartOrderEntry {
            line_idx,
            utterance_idx,
            speaker: utterance.main.speaker.as_str().to_string(),
            start_ms: bullet.timing.start_ms,
            end_ms: bullet.timing.end_ms,
            overlap: TranscribedOverlap::of(utterance),
        });
        entries.push(match mutability {
            BulletMutability::ReadOnly(_) => StartOrderSource::Kept(facts),
            BulletMutability::Revisable => {
                StartOrderSource::Revisable(StartOrderTarget { facts, utterance })
            }
        });
    }

    // The nearest kept entry AFTER each position. Kept bullets are never
    // stripped here, so the snapshot stays true for the whole pass.
    let mut next_kept: Vec<Option<Arc<StartOrderEntry>>> = vec![None; entries.len()];
    let mut following_kept = None;
    for position in (0..entries.len()).rev() {
        next_kept[position] = following_kept.clone();
        match &entries[position] {
            StartOrderSource::Kept(facts) => following_kept = Some(facts.clone()),
            StartOrderSource::Revisable(_) => {}
        }
    }

    // Decisions read immutable shared facts, so a stripped line
    // can be stripped the moment it is decided: the effect then carries the
    // measured word intervals the strip removed, which exist nowhere else
    // once the strip has run.
    let mut last_accepted: Option<Arc<StartOrderEntry>> = None;
    let mut last_kept: Option<Arc<StartOrderEntry>> = None;
    for (position, source) in entries.into_iter().enumerate() {
        let entry = source.facts().clone();
        match source {
            StartOrderSource::Kept(_) => {
                match last_kept
                    .as_ref()
                    .filter(|previous| entry.start_ms < previous.start_ms)
                {
                    Some(previous) => decisions.push(start_order_record(
                        &entry,
                        MonotonicityStrategy::KeptBulletLeftUnresolved,
                        format!(
                            "non_monotonic kept_main_bullet start_ms={} previous_start_ms={} \
                             previous_line_idx={} previous_speaker={}",
                            entry.start_ms, previous.start_ms, previous.line_idx, previous.speaker
                        ),
                    )),
                    None => {
                        last_kept = Some(entry.clone());
                        last_accepted = Some(entry.clone());
                    }
                }
            }
            StartOrderSource::Revisable(target) => {
                match target.classify(last_accepted.clone(), next_kept[position].clone()) {
                    StartOrderPosition::Regressing(regression) => {
                        let previous = regression.previous.clone();
                        let conflict = format!(
                            "non_monotonic start_ms={} previous_start_ms={} \
                             previous_line_idx={} previous_speaker={}",
                            entry.start_ms, previous.start_ms, previous.line_idx, previous.speaker
                        );
                        match regression.resolve() {
                            StartRegressionResolution::Stripped {
                                timing: StrippedTiming { words_dropped },
                                cause,
                            } => {
                                decisions.push(start_order_record(
                                    &entry,
                                    MonotonicityStrategy::TimingStripped,
                                    format!("{conflict} cause={}", cause.as_str()),
                                ));
                                effects.push(MonotonicityEffect::StartRegressionStripped {
                                    line_idx: entry.line_idx,
                                    utterance_idx: entry.utterance_idx,
                                    speaker: entry.speaker.clone(),
                                    start_ms: entry.start_ms,
                                    previous_start_ms: previous.start_ms,
                                    previous_line_idx: previous.line_idx,
                                    previous_utterance_idx: previous.utterance_idx,
                                    previous_speaker: previous.speaker.clone(),
                                    words_dropped,
                                });
                            }
                            StartRegressionResolution::RaisedToPrecedingStart {
                                words_dropped,
                                words_trimmed,
                            } => {
                                decisions.push(start_order_record(
                                    &entry,
                                    MonotonicityStrategy::StartRaisedToPrecedingStart,
                                    format!(
                                        "{conflict} overlap_marker=+< words_dropped={} \
                                         words_trimmed={}",
                                        words_dropped.len(),
                                        words_trimmed.len()
                                    ),
                                ));
                                effects.push(MonotonicityEffect::StartRegressionRaised {
                                    line_idx: entry.line_idx,
                                    utterance_idx: entry.utterance_idx,
                                    speaker: entry.speaker.clone(),
                                    start_ms: entry.start_ms,
                                    previous_start_ms: previous.start_ms,
                                    previous_line_idx: previous.line_idx,
                                    previous_utterance_idx: previous.utterance_idx,
                                    previous_speaker: previous.speaker.clone(),
                                    words_dropped,
                                    words_trimmed,
                                });
                                // The raised bullet starts exactly at the
                                // preceding start, so `last_accepted`, which
                                // already names that start, stays as it is;
                                // the snapshot entry still holds the old one.
                            }
                        }
                    }
                    StartOrderPosition::Yielding(yielding) => {
                        let ResolvedStartYield {
                            kept,
                            timing: StrippedTiming { words_dropped },
                        } = yielding.resolve();
                        decisions.push(start_order_record(
                            &entry,
                            MonotonicityStrategy::TimingStripped,
                            format!(
                                "yielded_to_kept_start start_ms={} kept_start_ms={} \
                                 kept_line_idx={} kept_speaker={}",
                                entry.start_ms, kept.start_ms, kept.line_idx, kept.speaker
                            ),
                        ));
                        effects.push(MonotonicityEffect::YieldedToKeptBullet {
                            line_idx: entry.line_idx,
                            utterance_idx: entry.utterance_idx,
                            speaker: entry.speaker.clone(),
                            kept_line_idx: kept.line_idx,
                            kept_utterance_idx: kept.utterance_idx,
                            kept_speaker: kept.speaker.clone(),
                            kept_boundary_ms: kept.start_ms,
                            outcome: KeptBulletYield::Stripped {
                                start_ms: entry.start_ms,
                                end_ms: entry.end_ms,
                                words_dropped,
                            },
                        });
                    }
                    StartOrderPosition::Accepted(facts) => last_accepted = Some(facts),
                }
            }
        }
    }
    Ok(())
}

/// The one record shape Pass 1 writes: every start-order decision needs
/// review, since each removed timing, moved a start off its measurement, or
/// left a conflict in place.
fn start_order_record(
    entry: &StartOrderEntry,
    strategy: batchalign_transform::decisions::MonotonicityStrategy,
    reason: String,
) -> batchalign_transform::decisions::DecisionRecord {
    batchalign_transform::decisions::DecisionRecord::new_and_trace(
        entry.line_idx,
        entry.speaker.clone(),
        batchalign_transform::decisions::DecisionStrategy::Monotonicity(strategy),
        reason,
        true,
    )
}
