//! Post-processing a single utterance's word timings: supply the ends an
//! engine did not report, rebalance collapsed words against their
//! neighbours, optionally bound the words to an authoritative utterance
//! bullet, and settle what survives into the AST.
//!
//! It does NOT update the utterance bullet. That is
//! `chat_ops::fa::update_utterance_bullet_with_boundary_policy`, called
//! by `chat_ops::fa::orchestrate` after this module has run; the doc comments
//! here claimed it as a fourth step until 2026-09-07.

use talkbank_model::alignment::helpers::{WordItem, WordItemMut, walk_words, walk_words_mut};
use talkbank_model::model::{
    Bullet, BulletSource, Utterance, UtteranceContent, Word, WordCategory,
};

use super::coordinates::{FileMs, Ms};
use super::get_word_timing;
use super::injection::InjectedTimings;
use super::origin::{ClampBound, Origin, ProvenanceTally};
use super::{
    DegenerateWordTiming, DroppedWordTimings, ExistingWorBoundaryPolicy, LAST_WORD_FALLBACK_MS,
    TimeSpan, WordEndPolicy, WordTiming,
};

/// A word's extent while post-processing is still deciding it, WITH the record
/// of how it got there.
///
/// # Why this exists
///
/// This module used to work on bare `TimeSpan`, deliberately: an extent under
/// construction may not satisfy `end > start` yet, so `WordTiming`'s invariant
/// cannot hold mid-computation. But dropping the invariant also dropped the
/// PROVENANCE, and that was not deliberate. Every pass below adjusts timings,
/// and a `TimeSpan` cannot say whether a value is what an engine measured, the
/// utterance bullet standing in for it, or a fabricated 500 ms; so the
/// distinction was destroyed here and reconstructed nowhere.
///
/// It derefs to the span, so reads are unchanged. Only the sites that CREATE a
/// value have to say what they did, which is the point.
///
/// # Why this is NOT merged into [`WordTiming`], though the fields match
///
/// Three review rounds have proposed collapsing the two on the grounds that
/// they hold the same three fields. They are a PHASE PAIR, and the fields are
/// the same on purpose: what differs is the invariant, which is the whole
/// reason both exist. A `WordTiming` guarantees `end > start`; a
/// `PendingTiming` does not, because the passes below move a boundary BETWEEN
/// two words and compute the new positions arithmetically. Those computations
/// are guarded, but the guard is arithmetic rather than proof, and
/// [`PendingTiming::settle`] is the single checked transition out.
///
/// Merging them costs one of two things and neither is worth it. Making
/// `WordTiming` the intermediate drops its invariant, which is what protects
/// the cache and every `%wor` tier. Making the intermediate carry the invariant
/// makes every `remeasured` call site fallible, so a repair that transiently
/// inverts a span could not be expressed at all and each site would grow a
/// `Result` for a case that `settle` already reports once.
///
/// Stated here rather than left for a fourth round to rediscover.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct PendingTiming {
    span: TimeSpan,
    model_score: Option<super::ModelAlignmentScore>,
    start_origin: Origin,
    end_origin: Origin,
}

/// What post-processing did, and what it learned.
///
/// # Why the tally rides along
///
/// The write-back lowers every `WordTiming` into a `Bullet`, which is two
/// integers, so the per-word provenance ends there and cannot be recovered
/// downstream. A summary CAN cross that boundary, and a caller that wants to
/// tell a reviewer "one of these sixteen timings was invented" has nowhere else
/// to get it. Returning it is the difference between provenance that reaches
/// the artifact and provenance that only reaches a cache.
pub struct PostprocessOutcome {
    /// Words that lost their timing, by reason.
    pub dropped: DroppedWordTimings,
    /// How the surviving timings were produced.
    pub provenance: ProvenanceTally,
}

impl From<DroppedWordTimings> for PostprocessOutcome {
    /// For the early returns, where no timing was examined at all.
    fn from(dropped: DroppedWordTimings) -> Self {
        Self {
            dropped,
            provenance: ProvenanceTally::default(),
        }
    }
}

/// One boundary of a word that a pass has just moved.
///
/// Handed to the closure in [`PendingTiming::remeasured`] so the new origin is
/// built from what actually happened rather than from what the call site
/// believed. Carrying `original` here is what fixed the two rebalance passes:
/// each captured a single `original` for two words whose boundaries moved in
/// opposite directions, so one of the two recorded the wrong end of the word.
pub(super) struct MovedBoundary {
    /// What this boundary meant before it moved. The new origin wraps it.
    pub was: Origin,
    /// Where the boundary was.
    pub original: FileMs,
    /// Where it is now.
    pub now: FileMs,
}

impl MovedBoundary {
    /// How far the boundary travelled, in whichever direction it went.
    ///
    /// A distance, so it is meaningful for a start pushed later and an end
    /// pulled earlier alike; the previous code could only express the latter
    /// and reported `Ms(0)` for the former.
    fn distance(&self) -> Ms {
        match self.original >= self.now {
            true => self.original.since(self.now),
            false => self.now.since(self.original),
        }
    }
}

impl PendingTiming {
    /// A timing this run produced, keeping the provenance it was given.
    fn from_aligned(timing: WordTiming) -> Self {
        let span = *timing;
        Self {
            model_score: timing.model_score(),
            start_origin: timing.start_origin().clone(),
            end_origin: timing.end_origin().clone(),
            span,
        }
    }

    /// A span lifted off the transcript, before any pass has touched it.
    fn from_transcript(span: TimeSpan) -> Self {
        Self {
            span,
            model_score: None,
            start_origin: Origin::TranscriptBullet,
            end_origin: Origin::TranscriptBullet,
        }
    }

    /// This word, re-timed by a pass, remembering what it was.
    ///
    /// The new origin WRAPS the old, so a value healed and then clamped still
    /// names the measurement underneath both.
    ///
    /// # Why the moved boundary is DERIVED and not stated
    ///
    /// This used to be `rederived(&self, ..)`, which rewrote `end_origin` and
    /// kept the start unconditionally, on the strength of a comment saying
    /// "every pass here moves a word's END". Three of its six call sites moved a
    /// START. So a boundary this program had just relocated kept the origin of
    /// the measurement it displaced, and `ProvenanceTally` counted the invented
    /// number as `Observed`: the exact defect the `Origin` type exists to
    /// prevent, one function below it. A second symptom of the same cause: each
    /// rebalance pass captured ONE `original` for two words whose boundaries
    /// moved in opposite directions, so half the recorded positions were the
    /// wrong end of the word.
    ///
    /// Comparing the new span against the old answers "which boundary moved"
    /// from the data, so no caller can get it wrong and no comment has to be
    /// believed. `why` is applied to each boundary that actually changed, and to
    /// neither when nothing did.
    fn remeasured(self, span: TimeSpan, why: impl Fn(MovedBoundary) -> Origin) -> Self {
        let was = self.span;
        Self {
            span,
            model_score: self.model_score,
            start_origin: match span.start_ms == was.start_ms {
                true => self.start_origin,
                false => why(MovedBoundary {
                    was: self.start_origin,
                    original: FileMs::new(was.start_ms),
                    now: FileMs::new(span.start_ms),
                }),
            },
            end_origin: match span.end_ms == was.end_ms {
                true => self.end_origin,
                false => why(MovedBoundary {
                    was: self.end_origin,
                    original: FileMs::new(was.end_ms),
                    now: FileMs::new(span.end_ms),
                }),
            },
        }
    }

    /// This word's END moved, and the pass is SUBSTITUTING its own account of
    /// what that boundary now means.
    ///
    /// Separate from [`PendingTiming::remeasured`] because these are different
    /// acts: `remeasured` WRAPS the previous origin, preserving the chain back
    /// to a measurement, while this REPLACES it, for passes that supply a value
    /// of their own (an inferred onset, an assumed duration) rather than
    /// adjusting the one that was there. It cannot touch the start, which is why
    /// it does not have to say which boundary it moved.
    ///
    /// Its three call sites previously passed `|_| ...` closures to `rederived`,
    /// ignoring the argument. A closure that discards its only parameter is the
    /// tell that the operation was the wrong shape.
    fn with_end(self, end_ms: u64, origin: Origin) -> Self {
        Self {
            span: TimeSpan::new(self.span.start_ms, end_ms),
            model_score: self.model_score,
            start_origin: self.start_origin,
            end_origin: origin,
        }
    }

    /// Cut to the utterance bullet's boundary, saying so.
    ///
    /// Named rather than written as `.max(..).min(..)` at the call site so that
    /// a word cut down to fit its utterance is distinguishable from one that
    /// already fitted. Which bound was applied is carried by
    /// [`ClampBound::UtteranceBullet`] in the resulting origin; this docstring
    /// used to assert that distinction in prose while the code wrote
    /// `ClampedToRecording`, and the prose was the only thing that knew.
    ///
    /// EITHER boundary may move here, since the caller passes a lower and an
    /// upper bound. `remeasured` marks whichever did.
    fn clamped_to_bullet(self, start_ms: u64, end_ms: u64) -> Option<Self> {
        let span = TimeSpan::new(start_ms, end_ms);
        match end_ms > start_ms {
            false => None,
            // When nothing was cut the span equals this one, so `remeasured`
            // wraps neither boundary and the origins pass through untouched.
            true => Some(self.remeasured(span, |moved| Origin::ClampedTo {
                bound: ClampBound::UtteranceBullet,
                overshoot: moved.distance(),
                original: moved.original,
                was: Box::new(moved.was),
            })),
        }
    }

    /// A working span for a test fixture, standing for one the transcript
    /// carried. Test-only, for the same reason `WordTiming::fixture` is.
    #[cfg(test)]
    pub(super) fn fixture(span: TimeSpan) -> Self {
        Self::from_transcript(span)
    }

    /// Settle the extent, or refuse. The phase transition out of this module.
    ///
    /// `Result`, not `Option`: the caller counts refusals, and
    /// `if settled.is_none() { count += 1 }` after building `settled` is
    /// recomputing what this function already knew.
    fn settle(self) -> Result<WordTiming, DegenerateWordTiming> {
        WordTiming::new(
            self.span.start_ms,
            self.span.end_ms,
            self.start_origin,
            self.end_origin,
        )
        .map(|timing| match self.model_score {
            Some(score) => timing.with_model_score(score),
            None => timing,
        })
        .ok_or(DegenerateWordTiming {
            start_ms: self.span.start_ms,
            end_ms: self.span.end_ms,
        })
    }
}

impl std::ops::Deref for PendingTiming {
    type Target = TimeSpan;

    fn deref(&self) -> &Self::Target {
        &self.span
    }
}

/// What clamping a transcript word's timing to a bound actually produced.
///
/// A word wholly past the bound cannot be trimmed to a positive extent, and
/// the old signature (`Option<WordTiming>`) answered `None` for that case
/// with no way for the caller to say what was lost. That is shape C (a total
/// function silently discarding information): the span this crate already
/// held before the clamp ran is a MEASURED fact (or, for a re-run word, the
/// transcript's own prior account of the word, which this crate treats as an
/// observation the same way `Origin::TranscriptBullet` does everywhere
/// else), and it went nowhere. `DroppedPastBound` carries it, so a caller
/// that counts drops can also say what each one was.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum WordClampOutcome {
    /// The word kept a positive extent after clamping.
    Trimmed(WordTiming),
    /// The word's start was already at or past the bound: no positive
    /// extent survives the cut. `measured` is the span as it sat in the
    /// transcript immediately before this clamp ran.
    DroppedPastBound {
        /// The extent that was lost.
        measured: TimeSpan,
    },
}

/// Clamp a word's timing, as it already sits in the transcript AST, to
/// `[start_ms, end_ms)`.
///
/// This is the one route by which a pass running AFTER this module has
/// already lowered timings to AST bullets (the cross-utterance monotonicity
/// pass in `orchestrate.rs`) may cut a word's timing. It seeds a
/// [`PendingTiming`] with [`Origin::TranscriptBullet`] (the same meaning that
/// origin already carries above: "a span lifted off the transcript, before
/// any pass has touched it", exactly true here, since as far as this later
/// pass is concerned the span is a value it found already written to the
/// AST, not a fresh measurement) and runs the clamp through the same
/// [`PendingTiming::clamped_to_bullet`] and [`PendingTiming::settle`] every
/// other clamp in this crate uses. The returned timing's end origin is
/// therefore a real `Origin::ClampedTo { bound: ClampBound::UtteranceBullet,
/// .. }` produced by the actual arithmetic, never a hand-written
/// substitute, and a clamp that would leave no positive extent reports
/// [`WordClampOutcome::DroppedPastBound`] with the span it lost, rather than
/// writing a degenerate span or discarding the fact entirely.
///
/// `talkbank-model`'s `Bullet` has no field for an `Origin` (it is two
/// integers plus source metadata), so the caller cannot carry a `Trimmed`
/// timing's origin into the CHAT file it writes back to; the value exists so
/// the CLAMP ITSELF runs through the real, invariant-checked route rather
/// than a second hand-rolled `.max()`/`.min()`.
pub(super) fn clamp_transcript_word_to_bullet(
    span: TimeSpan,
    start_ms: u64,
    end_ms: u64,
) -> WordClampOutcome {
    match PendingTiming::from_transcript(span).clamped_to_bullet(start_ms, end_ms) {
        None => WordClampOutcome::DroppedPastBound { measured: span },
        Some(pending) => match pending.settle() {
            Ok(timing) => WordClampOutcome::Trimmed(timing),
            Err(_degenerate) => WordClampOutcome::DroppedPastBound { measured: span },
        },
    }
}

/// Maximum internal gap (ms) that gap healing may collapse into the
/// previous word.
///
/// Small word-to-word gaps are a useful smoothing target, but multi-second
/// silences or mistracked spans should remain visible instead of turning one
/// word into a dominant 10-second token.
const MAX_HEALED_INTERNAL_GAP_MS: u64 = 1_000;
const MIN_HEALABLE_WORD_DURATION_MS: u64 = 40;
const MAX_HEALED_WORD_PROPORTION_NUMERATOR: u64 = 2;
const MAX_HEALED_WORD_PROPORTION_DENOMINATOR: u64 = 5;
/// When a rerun already has `%wor`, authoritative-bullet clamping normally
/// keeps FA word timings inside the previous utterance window. However, some
/// stale rerun bullets truncate the final word to a near-zero tail. Allow the
/// last timed word to heal a small overrun instead of preserving the collapse.
const MIN_HEALED_FINAL_WORD_DURATION_MS: u64 = 100;
const MAX_HEALED_FINAL_WORD_OVERRUN_MS: u64 = 500;

/// Post-process one utterance's word timings, in this order.
///
/// 1. Under `WordEndPolicy`'s healing and derived-end settings: give the LAST
///    timed word an end (the utterance bullet if there is one, otherwise an
///    assumed duration), extend each earlier word to its successor's onset
///    where the internal gap is plausibly small, and rebalance a lexical word
///    that collapsed to near-zero duration against its neighbour.
/// 2. Bound the words to the utterance bullet, but ONLY when that bullet is
///    `BulletSource::Authoritative`, the utterance already has a `%wor` tier,
///    and the boundary policy is `Preserve`. The long comment at that block
///    says why each condition is load-bearing; "bound all word times within
///    utterance bullet range" was the previous wording here and is true of
///    none of the other cases.
/// 3. Settle each timing, dropping any with no positive extent.
/// 4. Write the survivors back to the AST and tally how they were produced.
///
/// It does NOT touch the utterance's own bullet; see the module doc.
///
/// Returns the words that lost their timing, by cause, and the provenance
/// tally of the ones that survived.
pub fn postprocess_utterance_timings(
    utterance: &mut Utterance,
    policy: WordEndPolicy,
    injected: &InjectedTimings,
) -> PostprocessOutcome {
    postprocess_utterance_timings_with_boundary_policy(
        utterance,
        policy,
        ExistingWorBoundaryPolicy::Preserve,
        injected,
    )
}

/// Post-process with an explicit policy for boundaries inherited from `%wor`.
pub fn postprocess_utterance_timings_with_boundary_policy(
    utterance: &mut Utterance,
    policy: WordEndPolicy,
    existing_wor_boundaries: ExistingWorBoundaryPolicy,
    injected: &InjectedTimings,
) -> PostprocessOutcome {
    // Origins come from the VALUES, never from the bullets they were written
    // into, which are two integers and cannot carry one. This was a two-variant
    // `TimingSeed` until 2026-08-15, whose second variant read the bullets back
    // and stamped every one `TranscriptBullet`; production never selected it,
    // so an enum with one reachable variant was standing in for a plain
    // argument. Fixtures that need transcript spans build them through
    // `InjectedTimings::from_transcript`, which is `#[cfg(test)]`.
    let mut word_timings: Vec<Option<PendingTiming>> = injected
        .as_slice()
        .iter()
        .map(|timing| timing.clone().map(PendingTiming::from_aligned))
        .collect();
    let mut word_is_compound_filler: Vec<bool> = Vec::new();
    collect_compound_filler_flags(
        &utterance.main.content.content,
        &mut word_is_compound_filler,
    );
    let mut word_is_filler: Vec<bool> = Vec::new();
    collect_filler_flags(&utterance.main.content.content, &mut word_is_filler);

    let mut dropped = DroppedWordTimings::default();
    if word_timings.is_empty() {
        return PostprocessOutcome::from(dropped);
    }
    debug_assert_eq!(word_timings.len(), word_is_compound_filler.len());
    debug_assert_eq!(word_timings.len(), word_is_filler.len());
    let utterance_span_ms = utterance
        .main
        .content
        .bullet
        .as_ref()
        .map(|bullet| bullet.timing.end_ms.saturating_sub(bullet.timing.start_ms))
        .or_else(|| {
            let first = word_timings
                .iter()
                .find_map(|timing| timing.as_ref().map(|span| span.start_ms))?;
            let last = word_timings
                .iter()
                .rev()
                .find_map(|timing| timing.as_ref().map(|span| span.end_ms))?;
            Some(last.saturating_sub(first))
        });

    // Under `Heal`: set each word's end_ms to the next word's start_ms.
    // Uses a backward pass (O(w)) instead of per-word forward scan (O(w²)).
    if policy.heals() {
        let n = word_timings.len();

        // Last timed word: it has no successor, so its end is the one the
        // engine could not supply. The utterance bullet is a better answer
        // than any constant when there is one.
        //
        // A measured end is left alone: only a derived one may be replaced,
        // and only ever extended, never shortened.
        // Index and scalars in ONE pass through a borrow: the value itself is
        // moved out only on the branch that writes it back, so the common
        // no-change path copies nothing. Reading them together also removes the
        // "`rposition` proved this is `Some`" step that a separate lookup needs
        // and that has no honest answer if it ever fails.
        if policy.ends_are_derived()
            && let Some((idx, span_start_ms, span_end_ms)) = word_timings
                .iter()
                .enumerate()
                .rev()
                .find_map(|(idx, slot)| slot.as_ref().map(|s| (idx, s.start_ms, s.end_ms)))
        {
            // Two genuinely different answers, and which one was used is a
            // fact about the number: the utterance bullet is a boundary a human
            // or an earlier pass placed, the constant is invented outright.
            // `unwrap_or` made them the same integer.
            let (end_ms, invented) = match utterance
                .main
                .content
                .bullet
                .as_ref()
                .map(|bullet| bullet.timing.end_ms)
                .filter(|utt_end| *utt_end > span_start_ms)
            {
                Some(utt_end) => (
                    utt_end,
                    Origin::InheritedFromNeighbour {
                        from: FileMs::new(utt_end),
                    },
                ),
                None => (
                    span_start_ms + LAST_WORD_FALLBACK_MS,
                    Origin::FallbackDuration {
                        assumed: Ms(LAST_WORD_FALLBACK_MS),
                    },
                ),
            };
            // `with_end`, not `remeasured`: this pass SUBSTITUTES its own
            // account of the end rather than adjusting the one that was there.
            if end_ms > span_end_ms
                && let Some(span) = word_timings[idx].take()
            {
                word_timings[idx] = Some(span.with_end(end_ms, invented));
            }
        }

        // Backward pass: track the next timed word's start_ms and propagate it
        // as the current word's end_ms only for plausibly small internal gaps.
        let mut next_start: Option<u64> = None;
        let mut next_is_filler = false;
        for i in (0..n).rev() {
            // Scalars through a borrow. This used to `.clone()` EVERY word on
            // every run to read three integers, and the clone is two `Origin`s,
            // one of which is commonly a `Box`ed chain by this point. The value
            // is moved out with `take()` only in the two branches below that
            // write it back.
            if let Some((span_start_ms, span_end_ms)) =
                word_timings[i].as_ref().map(|s| (s.start_ms, s.end_ms))
            {
                if let Some(ns) = next_start {
                    let gap_ms = ns.saturating_sub(span_end_ms);
                    let bridged_duration = ns.saturating_sub(span_start_ms);
                    let lexical_to_filler_bridge_is_plausible = if next_is_filler
                        && !word_is_filler[i]
                    {
                        span_end_ms.saturating_sub(span_start_ms) < MIN_HEALABLE_WORD_DURATION_MS
                            && bridged_duration_stays_within_proportion_cap(
                                utterance_span_ms,
                                bridged_duration,
                            )
                    } else {
                        true
                    };
                    let filler_bridge_is_plausible = if word_is_filler[i] {
                        bridged_duration_stays_within_proportion_cap(
                            utterance_span_ms,
                            bridged_duration,
                        )
                    } else {
                        true
                    };
                    let should_fill_gap = if next_is_filler {
                        lexical_to_filler_bridge_is_plausible && filler_bridge_is_plausible
                    } else {
                        gap_ms <= MAX_HEALED_INTERNAL_GAP_MS && filler_bridge_is_plausible
                    };
                    if should_fill_gap && !word_is_compound_filler[i] {
                        // Refusing here is not the same as letting the
                        // write-back drop it: this KEEPS the measured span,
                        // where the boundary would discard the word entirely.
                        // A non-monotonic response asks for exactly this, and
                        // no gate above notices, because `gap_ms` saturates to
                        // zero when the next word begins before this one ends.
                        // The guard is load-bearing and is NOT the same as
                        // letting the write-back drop it: refusing here KEEPS
                        // the measured span, where a backwards heal would
                        // replace it with an end preceding its own start. A
                        // non-monotonic engine response asks for exactly that,
                        // and no gate above notices, because `gap_ms` saturates
                        // to zero when the next word begins before this one
                        // ends.
                        if ns > span_start_ms
                            && let Some(span) = word_timings[i].take()
                        {
                            word_timings[i] = Some(span.with_end(ns, Origin::DerivedFromNextOnset));
                        }
                    } else if span_start_ms == span_end_ms {
                        // Reachable only if an engine reports `start == end`
                        // for a word that HAS a successor: the onset-only
                        // parser no longer emits that shape, but a
                        // word-interval model could. The next onset caps the
                        // fallback, so this is a floor rather than an invented
                        // duration.
                        let assumed = span_start_ms + LAST_WORD_FALLBACK_MS;
                        let capped_end = assumed.min(ns);
                        let invented = Origin::FallbackDuration {
                            assumed: Ms(LAST_WORD_FALLBACK_MS),
                        };
                        // The cap is itself an adjustment, and it wraps the
                        // assumption rather than replacing it, so the value
                        // still says it was invented before it was cut. Chosen
                        // as an ORIGIN rather than as a boxed closure: both arms
                        // ignored the previous origin, so the trait object
                        // bought nothing but two allocations per word.
                        let origin = match capped_end < assumed {
                            true => Origin::ClampedTo {
                                bound: ClampBound::NextOnset,
                                was: Box::new(invented),
                                original: FileMs::new(assumed),
                                overshoot: Ms(assumed - capped_end),
                            },
                            false => invented,
                        };
                        if let Some(span) = word_timings[i].take() {
                            word_timings[i] = Some(span.with_end(capped_end, origin));
                        }
                    }
                }
                next_start = Some(span_start_ms);
                next_is_filler = word_is_filler[i];
            }
        }

        rebalance_near_zero_lexical_words_from_following_spans(&mut word_timings, &word_is_filler);
        rebalance_near_zero_lexical_words_from_preceding_spans(&mut word_timings, &word_is_filler);
    }

    // Bound by utterance bullet range, but ONLY when both conditions hold:
    //
    // 1. The bullet is `BulletSource::Authoritative` (not a runtime UTR hint).
    //    UTR-hinted bullets (`BulletSource::Utr`) are provisional estimates from
    //    ASR token timestamps.  They can be much narrower than the actual speech
    //    window: e.g., Rev.AI may produce a 220ms hint for a 3-second utterance
    //    when it only recognised the first word.  Clamping FA word timings to a
    //    UTR hint would drop every word beyond the first.
    //
    // 2. The utterance already has a `%wor` tier (i.e., this is a RE-alignment,
    //    not a first-time alignment).
    //    After `transcribe` + `utseg`, utterance bullets are ASR-derived (from
    //    UTR token matching) and are serialized as `Authoritative` (BulletSource
    //    is not persisted in CHAT text).  These bullets can be as narrow as the
    //    ASR-matched span for one word (e.g., 220ms for a 3-second sentence when
    //    Rev.AI only matched the first word).  FA, given a wider group audio
    //    window, correctly aligns all words, but clamping to the narrow bullet
    //    would drop all but the first, breaking the output.
    //    The `%wor` tier is only present after a previous FA run, which means the
    //    utterance bullet was established by FA from word timings and is wide
    //    enough to cover the speech.  That is the only case where clamping is safe.
    //
    // Self-healing: `update_utterance_bullet` overwrites UTR hints with the FA
    // word span after postprocess.  Clamping to a narrow UTR/ASR bullet before
    // that overwrite would prevent the self-healing from ever running.
    let has_fa_wor = utterance.wor_tier().is_some();
    if let Some(ref bullet) = utterance.main.content.bullet
        && bullet.source == BulletSource::Authoritative
        && has_fa_wor
        && existing_wor_boundaries == ExistingWorBoundaryPolicy::Preserve
    {
        let utt_start = bullet.timing.start_ms;
        let utt_end = bullet.timing.end_ms;
        let last_timed_idx = word_timings.iter().rposition(|timing| timing.is_some());

        for (idx, timing) in word_timings.iter_mut().enumerate() {
            if let Some(span) = timing {
                let clamped_start = span.start_ms.max(utt_start);
                let mut clamped_end = span.end_ms.min(utt_end);
                if Some(idx) == last_timed_idx
                    && clamped_end < span.end_ms
                    && clamped_end.saturating_sub(clamped_start) < MIN_HEALED_FINAL_WORD_DURATION_MS
                {
                    let overrun_ms = span.end_ms.saturating_sub(utt_end);
                    if overrun_ms <= MAX_HEALED_FINAL_WORD_OVERRUN_MS {
                        clamped_end = span.end_ms;
                    }
                }
                // Moved out rather than borrowed: `clamped_to_bullet` consumes
                // the value so it can hand each moved boundary's origin to the
                // new one without cloning either.
                *timing = match timing
                    .take()
                    .and_then(|span| span.clamped_to_bullet(clamped_start, clamped_end))
                {
                    Some(clamped) => Some(clamped),
                    None => {
                        // Counted, not logged per word. The count is RETURNED in
                        // `DroppedWordTimings`, which is the contract; a `warn!`
                        // inside the loop added nothing a caller could branch on
                        // and emitted one line per dropped word. Its aggregate
                        // now sits beside `without_extent` below, so the two drop
                        // reasons are reported the same way instead of one being
                        // per-word noise and the other a single summary.
                        dropped.clamped_to_utterance_boundary += 1;
                        None
                    }
                };
            }
        }
    }

    // Write timings back to the AST.
    //
    // This is the phase transition: everything above works on `TimeSpan`,
    // which carries no invariant because an extent under construction may not
    // have one yet. Nothing without a real extent gets past here, so a word's
    // bullet cannot hold a zero-length or backwards span however the
    // intermediate arithmetic went. A span that fails is dropped rather than
    // written, which is the same outcome the clamping block already chose for
    // the case it happens to cover.
    // `into_iter`, not `iter`: `word_timings` is not read after this, and
    // settling by value hands the two origins straight to the `WordTiming`
    // instead of deep-copying a `Box`ed provenance chain per word.
    let validated: Vec<Option<WordTiming>> = word_timings
        .into_iter()
        .map(|timing| {
            timing.and_then(|pending| {
                pending
                    .settle()
                    .inspect_err(|_| dropped.without_extent += 1)
                    .ok()
            })
        })
        .collect();

    // What survived, and how each one came to be. Counted here because this is
    // the last point the origins exist: the write below lowers them into
    // `Bullet`, which is two integers.
    let mut provenance = ProvenanceTally::default();
    // BOTH boundaries are counted: a word has two, they are routinely produced
    // differently, and counting one was the loss this pass exists to stop.
    for timing in validated.iter().flatten() {
        provenance.record(timing.start_origin());
        provenance.record(timing.end_origin());
    }
    if dropped.without_extent > 0 {
        tracing::warn!(
            without_extent = dropped.without_extent,
            "post-processing produced word timings with no extent; those words are left unaligned"
        );
    }
    if dropped.clamped_to_utterance_boundary > 0 {
        tracing::warn!(
            clamped_to_utterance_boundary = dropped.clamped_to_utterance_boundary,
            "word timings dropped: clamping to the utterance boundary left no extent"
        );
    }

    let mut idx = 0;
    set_word_timings(
        utterance.main.content.content.as_mut_slice(),
        &validated,
        &mut idx,
    );

    PostprocessOutcome {
        dropped,
        provenance,
    }
}

/// Wrap a boundary's origin to record that a rebalance pass moved it.
///
/// Shared by both rebalance passes, which previously wrote this closure twice.
/// It takes no `original`: [`MovedBoundary`] carries the position of whichever
/// boundary actually moved, which is exactly what the two hand-captured copies
/// got wrong.
///
/// Named `repaired_for_order` until 2026-09-07, after the origin variant it
/// builds, which was itself named for an ordering repair in
/// `chat_ops::fa::repair` that produces no origin at all. What actually
/// happens here is a rebalance against a neighbour; the name now says that on
/// both sides.
fn rebalanced_with_neighbour(moved: MovedBoundary) -> Origin {
    Origin::RebalancedWithNeighbour {
        was: Box::new(moved.was),
        original: moved.original,
    }
}

fn rebalance_near_zero_lexical_words_from_following_spans(
    word_timings: &mut [Option<PendingTiming>],
    word_is_filler: &[bool],
) {
    debug_assert_eq!(word_timings.len(), word_is_filler.len());

    for i in 0..word_timings.len().saturating_sub(1) {
        // Borrowed, not cloned: the guards below reject almost every pair, and
        // an origin a previous pass nested carries a `Box`, so a discarded clone
        // here is a heap deep-copy. Work happens only on the hit.
        let (Some(current), Some(next)) = (&word_timings[i], &word_timings[i + 1]) else {
            continue;
        };
        if word_is_filler[i] {
            continue;
        }
        if current.duration_ms() >= MIN_HEALABLE_WORD_DURATION_MS {
            continue;
        }
        if current.end_ms != next.start_ms {
            continue;
        }

        let needed_ms = MIN_HEALABLE_WORD_DURATION_MS - current.duration_ms();
        if next.duration_ms() < needed_ms + MIN_HEALABLE_WORD_DURATION_MS {
            continue;
        }

        // Both words are re-timed: the boundary between them MOVED, so neither
        // side of it is any longer what was measured. `remeasured` marks
        // whichever boundary each word actually lost, which is the fix: the old
        // shared closure derived ONE `original` from ONE field and applied it to
        // both words, so `next`, whose START moved, recorded its END's position
        // and kept its start's measured origin unchanged.
        let new_boundary = next.start_ms + needed_ms;
        let current_span = TimeSpan::new(current.start_ms, new_boundary);
        let next_span = TimeSpan::new(new_boundary, next.end_ms);

        // The borrows end above. Taken in separate statements because two
        // `&mut` into one slice in a single expression do not coexist.
        let taken_current = word_timings[i].take();
        let taken_next = word_timings[i + 1].take();
        match (taken_current, taken_next) {
            (Some(current), Some(next)) => {
                word_timings[i] = Some(current.remeasured(current_span, rebalanced_with_neighbour));
                word_timings[i + 1] = Some(next.remeasured(next_span, rebalanced_with_neighbour));
            }
            // The guards above proved both slots occupied. Written out rather
            // than unwrapped, and it puts back what it took, so a change to
            // those guards cannot silently discard a word's timing.
            (current, next) => {
                word_timings[i] = current;
                word_timings[i + 1] = next;
            }
        }
    }
}

fn rebalance_near_zero_lexical_words_from_preceding_spans(
    word_timings: &mut [Option<PendingTiming>],
    word_is_filler: &[bool],
) {
    debug_assert_eq!(word_timings.len(), word_is_filler.len());

    for i in 1..word_timings.len() {
        // Same: guard on borrows, clone only when the rebalance actually fires.
        let (Some(previous), Some(current)) = (&word_timings[i - 1], &word_timings[i]) else {
            continue;
        };
        if word_is_filler[i] {
            continue;
        }
        if current.duration_ms() >= MIN_HEALABLE_WORD_DURATION_MS {
            continue;
        }
        if previous.end_ms != current.start_ms {
            continue;
        }

        let needed_ms = MIN_HEALABLE_WORD_DURATION_MS - current.duration_ms();
        if previous.duration_ms() < needed_ms + MIN_HEALABLE_WORD_DURATION_MS {
            continue;
        }

        // Same move, borrowing from the preceding word instead. This pass had
        // the mirror of the other's defect: its shared closure recorded
        // `t.start_ms` for both words, so `previous`, whose END moved, recorded
        // its start's position.
        let new_boundary = current.start_ms - needed_ms;
        let previous_span = TimeSpan::new(previous.start_ms, new_boundary);
        let current_span = TimeSpan::new(new_boundary, current.end_ms);

        let taken_previous = word_timings[i - 1].take();
        let taken_current = word_timings[i].take();
        match (taken_previous, taken_current) {
            (Some(previous), Some(current)) => {
                word_timings[i - 1] =
                    Some(previous.remeasured(previous_span, rebalanced_with_neighbour));
                word_timings[i] = Some(current.remeasured(current_span, rebalanced_with_neighbour));
            }
            // As above: put back what was taken rather than unwrap.
            (previous, current) => {
                word_timings[i - 1] = previous;
                word_timings[i] = current;
            }
        }
    }
}

fn bridged_duration_stays_within_proportion_cap(
    utterance_span_ms: Option<u64>,
    bridged_duration_ms: u64,
) -> bool {
    utterance_span_ms.is_some_and(|utterance_span_ms| {
        bridged_duration_ms.saturating_mul(MAX_HEALED_WORD_PROPORTION_DENOMINATOR)
            <= utterance_span_ms.saturating_mul(MAX_HEALED_WORD_PROPORTION_NUMERATOR)
    })
}

/// Collect current word timings in document order.
///
/// Visits ALL words (no alignability filter). For replaced words, only the
/// original (spoken) word's timing is collected.
pub(super) fn collect_word_timings(
    content: &[UtteranceContent],
    out: &mut Vec<Option<PendingTiming>>,
) {
    // domain=None: recurse into all groups unconditionally
    walk_words(content, None, &mut |leaf| match leaf {
        WordItem::Word(word) => {
            out.push(get_word_timing(word).map(PendingTiming::from_transcript));
        }
        WordItem::ReplacedWord(replaced) => {
            out.push(get_word_timing(&replaced.word).map(PendingTiming::from_transcript));
        }
        WordItem::Separator(_) => {}
    });
}

/// The spans a transcript already carries, in the same document order as
/// [`collect_word_timings`].
///
/// Test-only: its one caller is `InjectedTimings::from_transcript`, which is
/// how fixtures seed post-processing now that `TimingSeed::FromTranscript` is
/// gone.
#[cfg(test)]
pub(super) fn collect_transcript_spans(
    content: &[UtteranceContent],
    out: &mut Vec<Option<super::TimeSpan>>,
) {
    walk_words(content, None, &mut |leaf| match leaf {
        WordItem::Word(word) => out.push(get_word_timing(word)),
        WordItem::ReplacedWord(replaced) => out.push(get_word_timing(&replaced.word)),
        WordItem::Separator(_) => {}
    });
}

fn collect_compound_filler_flags(content: &[UtteranceContent], out: &mut Vec<bool>) {
    walk_words(content, None, &mut |leaf| match leaf {
        WordItem::Word(word) => {
            out.push(super::split_compound_filler(word).len() > 1);
        }
        WordItem::ReplacedWord(replaced) => {
            out.push(super::split_compound_filler(&replaced.word).len() > 1);
        }
        WordItem::Separator(_) => {}
    });
}

fn collect_filler_flags(content: &[UtteranceContent], out: &mut Vec<bool>) {
    walk_words(content, None, &mut |leaf| match leaf {
        WordItem::Word(word) => {
            out.push(word.category == Some(WordCategory::Filler));
        }
        WordItem::ReplacedWord(replaced) => {
            out.push(replaced.word.category == Some(WordCategory::Filler));
        }
        WordItem::Separator(_) => {}
    });
}

/// Write timings back into word AST nodes.
///
/// Visits ALL words in document order (same order as `collect_word_timings`).
/// For replaced words, sets timing on the original (spoken) word only.
fn set_word_timings(
    content: &mut [UtteranceContent],
    timings: &[Option<WordTiming>],
    idx: &mut usize,
) {
    // domain=None: recurse into all groups unconditionally
    walk_words_mut(content, None, &mut |leaf| match leaf {
        WordItemMut::Word(word) => {
            set_word_timing(word, timings, idx);
        }
        WordItemMut::ReplacedWord(replaced) => {
            set_word_timing(&mut replaced.word, timings, idx);
        }
        WordItemMut::Separator(_) => {}
    });
}

fn set_word_timing(word: &mut Word, timings: &[Option<WordTiming>], idx: &mut usize) {
    if *idx < timings.len() {
        match &timings[*idx] {
            Some(span) => {
                word.inline_bullet = Some(Bullet::new(span.start_ms, span.end_ms));
            }
            None => {
                word.inline_bullet = None;
            }
        }
    }
    *idx += 1;
}

#[cfg(test)]
mod moved_boundary_tests {
    use super::*;
    use crate::chat_ops::fa::ModelAlignmentScore;

    fn measured(engine: &'static str) -> Origin {
        Origin::EngineMeasured {
            engine: super::super::origin::EngineId::new(engine),
        }
    }

    /// A word whose span is `[start, end)` with both boundaries measured.
    fn word(start_ms: u64, end_ms: u64) -> PendingTiming {
        PendingTiming {
            span: TimeSpan::new(start_ms, end_ms),
            model_score: None,
            start_origin: measured("wav2vec_fa"),
            end_origin: measured("wav2vec_fa"),
        }
    }

    #[test]
    fn a_pass_that_moves_only_the_start_marks_the_start_and_not_the_end() {
        // The defect this replaced `rederived` to fix. That function rewrote
        // `end_origin` unconditionally and kept `start_origin`, so a boundary
        // the program had just relocated kept the origin of the measurement it
        // displaced, and the untouched end was stamped with a clamp that never
        // happened.
        let moved_start = word(1_000, 2_000).remeasured(TimeSpan::new(1_200, 2_000), |m| {
            Origin::RebalancedWithNeighbour {
                was: Box::new(m.was),
                original: m.original,
            }
        });

        // The start moved, so it is no longer an observation, and it names
        // where it came FROM.
        assert!(!moved_start.start_origin.is_observation());
        assert_eq!(
            moved_start.start_origin,
            Origin::RebalancedWithNeighbour {
                was: Box::new(measured("wav2vec_fa")),
                original: FileMs::new(1_000),
            }
        );
        // The end did not move, so it is untouched and still measured.
        assert_eq!(moved_start.end_origin, measured("wav2vec_fa"));
        assert!(moved_start.end_origin.is_observation());
    }

    #[test]
    fn postprocessing_preserves_the_model_score_while_marking_rebalances() {
        let score = ModelAlignmentScore::try_from_f64(0.73).unwrap();
        let aligned = WordTiming::new(100, 300, measured("wav2vec_fa"), measured("wav2vec_fa"))
            .unwrap()
            .with_model_score(score);

        let settled = PendingTiming::from_aligned(aligned)
            .remeasured(TimeSpan::new(120, 300), |moved| {
                Origin::RebalancedWithNeighbour {
                    was: Box::new(moved.was),
                    original: moved.original,
                }
            })
            .settle()
            .unwrap();

        assert_eq!(settled.model_score(), Some(score));
        assert!(!settled.start_origin().is_observation());
    }

    #[test]
    fn a_pass_that_moves_nothing_wraps_nothing() {
        let unchanged = word(1_000, 2_000)
            .remeasured(TimeSpan::new(1_000, 2_000), |_| Origin::TranscriptBullet);
        assert_eq!(unchanged.start_origin, measured("wav2vec_fa"));
        assert_eq!(unchanged.end_origin, measured("wav2vec_fa"));
    }

    #[test]
    fn each_rebalanced_word_records_the_boundary_it_actually_lost() {
        // Both rebalance passes moved a boundary of two adjacent words in
        // OPPOSITE directions while deriving one `original` from one field, so
        // one of the pair always recorded the wrong end of itself.
        //
        // `current` is under the healable floor and abuts `next`, which is long
        // enough to donate. The shared boundary at 2_000 moves later.
        let mut timings = vec![Some(word(1_980, 2_000)), Some(word(2_000, 3_000))];
        rebalance_near_zero_lexical_words_from_following_spans(&mut timings, &[false, false]);

        let (current, next) = (
            timings[0].as_ref().expect("current survives"),
            timings[1].as_ref().expect("next survives"),
        );
        // They still meet, at the new boundary.
        assert_eq!(current.span.end_ms, next.span.start_ms);
        let boundary = current.span.end_ms;
        assert!(boundary > 2_000, "the boundary should have moved later");

        // `current` lost its END: that is what must be recorded, at its OWN
        // former position, and its start must be untouched.
        assert_eq!(current.start_origin, measured("wav2vec_fa"));
        assert_eq!(
            current.end_origin,
            Origin::RebalancedWithNeighbour {
                was: Box::new(measured("wav2vec_fa")),
                original: FileMs::new(2_000),
            }
        );

        // `next` lost its START. Under the old code this word's END was stamped
        // instead, with `next.end_ms` as the position, and its moved start kept
        // a measured origin that `ProvenanceTally` then counted as observed.
        assert_eq!(next.end_origin, measured("wav2vec_fa"));
        assert_eq!(
            next.start_origin,
            Origin::RebalancedWithNeighbour {
                was: Box::new(measured("wav2vec_fa")),
                original: FileMs::new(2_000),
            }
        );
    }

    #[test]
    fn a_start_pushed_later_reports_the_distance_it_travelled() {
        // `overshoot` was computed as `end - new_end`, which is zero when only
        // the start moves, so a real adjustment reported itself as a no-op.
        let clamped = word(1_000, 2_000)
            .clamped_to_bullet(1_300, 2_000)
            .expect("still has extent");
        assert_eq!(
            clamped.start_origin,
            Origin::ClampedTo {
                bound: ClampBound::UtteranceBullet,
                overshoot: Ms(300),
                original: FileMs::new(1_000),
                was: Box::new(measured("wav2vec_fa")),
            }
        );
        assert_eq!(clamped.end_origin, measured("wav2vec_fa"));
    }
}

#[cfg(test)]
mod clamp_transcript_word_to_bullet_tests {
    use super::*;

    /// A word that starts at or past the bound cannot be trimmed to a
    /// positive extent, and the extent it HAD is a fact this module already
    /// possessed before the clamp ran. Throwing it away without recording it
    /// is shape C (a total function silently discarding information); this
    /// pins the cure: `DroppedPastBound` carries exactly the span the
    /// transcript held.
    #[test]
    fn a_word_wholly_past_the_bound_reports_dropped_with_its_measured_span() {
        let outcome = clamp_transcript_word_to_bullet(TimeSpan::new(4_200, 4_800), 4_200, 4_000);

        assert_eq!(
            outcome,
            WordClampOutcome::DroppedPastBound {
                measured: TimeSpan::new(4_200, 4_800),
            }
        );
    }

    /// A word straddling the bound keeps a positive extent, cut to it: the
    /// ordinary case, unchanged by the type carrying the other one.
    #[test]
    fn a_word_straddling_the_bound_is_trimmed() {
        let outcome = clamp_transcript_word_to_bullet(TimeSpan::new(3_000, 5_000), 3_000, 4_000);

        let WordClampOutcome::Trimmed(timing) = outcome else {
            panic!("a straddling word must trim, not drop: {outcome:?}");
        };
        assert_eq!(timing.start_ms, 3_000);
        assert_eq!(timing.end_ms, 4_000);
    }

    /// A word whose start already sits exactly at the bound has no positive
    /// extent left either, same as one further past it: the boundary case
    /// for the `end_ms > start_ms` cut.
    #[test]
    fn a_word_starting_exactly_at_the_bound_is_dropped_not_zero_length() {
        let outcome = clamp_transcript_word_to_bullet(TimeSpan::new(4_000, 4_500), 4_000, 4_000);

        assert_eq!(
            outcome,
            WordClampOutcome::DroppedPastBound {
                measured: TimeSpan::new(4_000, 4_500),
            }
        );
    }
}
