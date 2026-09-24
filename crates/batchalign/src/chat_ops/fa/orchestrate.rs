//! Full-file orchestration: apply FA results, finalize timing, retain evidence.

use std::collections::HashMap;

use talkbank_model::UtteranceIdx;
use talkbank_model::alignment::helpers::{WordItem, WordItemMut, walk_words, walk_words_mut};
use talkbank_model::alignment::{
    WorTimingBinding, WorTimingCorrespondence, WorTimingSequence, assess_wor_timing_sequence,
    bind_wor_timing, corroborate_wor_timing,
};
use talkbank_model::model::dependent_tier::{WorItem, WorTier};
use talkbank_model::model::{
    BracketedItem, Bullet, ChatFile, DependentTier, Line, Utterance, UtteranceContent, Word,
};
use talkbank_model::model::{BracketedItems, TierContentItems};

#[cfg(test)]
use super::WordEndPolicy;
use super::injection::inject_timings_for_utterance;
use super::main_bullets::{
    BulletMutability, FaProjection, GivenMutability, ImposedBullets, KeptBullet, KeptBulletError,
    KeptBulletsHeld, MainBulletAuthority,
};
use super::origin::Origin;
use super::postprocess::postprocess_utterance_timings_with_boundary_policy;
use super::repair::{BulletRepairPolicy, RepairStats, repair_bullets_under};
use super::{EndOverlapPolicy, ExistingWorBoundaryPolicy};
use super::{
    FaGroup, TimeSpan, WordTiming, add_wor_tier, count_alignable_main_words, get_utterance_mut,
    update_utterance_bullet_with_boundary_policy,
};

/// What the write phase should do about `%wor`, once monotonicity has
/// resolved same-speaker overlaps.
///
/// Replaces a `write_wor: bool` + `touched: Vec<UtteranceIdx>` pair that
/// admitted a state nothing ever read: a non-empty `touched` with
/// `write_wor` false. That state was never CONSTRUCTED wrongly by the one
/// call site that built both fields together, but every reader still had to
/// know "touched only matters when write_wor is true" from a comment; here
/// there is only one thing to read, and its shape says what it means.
///
/// `%wor` generation used to happen inline, per utterance, in the same loop
/// that injects and postprocesses timings, BEFORE monotonicity ran over the
/// whole file. That ordering was the defect the `Pending` variant exists to
/// close: a same-speaker overlap resolved by cutting a word (see
/// [`EndOverlapResolution::InterleavedWords`]) would leave an
/// already-written `%wor` stale, holding the pre-cut span. Carrying the
/// request here, instead of acting on it immediately, makes writing `%wor` a
/// step of [`FaApplied::then_enforce_monotonicity`] rather than of the
/// per-utterance loop.
#[derive(Debug, Clone)]
pub(super) enum WorPlan {
    /// No `%wor` write requested for this run.
    Suppressed,
    /// `%wor` should be (re)written for these utterances, once monotonicity
    /// resolves. Two disjoint reasons an utterance ends up in this list, and
    /// both are legitimate: THIS run injected fresh timings into it (the
    /// `apply_fa_results_with_projection_policy` loop), or a caller
    /// separately refreshed it from reusable `%wor` and folded it in via
    /// [`FaApplied::also_touched`] so the SAME write phase covers it too.
    /// An utterance neither of these produced has nothing new to reflect
    /// and is never in this list.
    Pending(Vec<UtteranceIdx>),
}

impl WorPlan {
    fn new(write_wor: bool, touched: Vec<UtteranceIdx>) -> Self {
        match write_wor {
            true => Self::Pending(touched),
            false => Self::Suppressed,
        }
    }

    /// Fold in utterances a caller separately refreshed and wants covered
    /// by this same write phase. A no-op on `Suppressed`: a run that asked
    /// for no `%wor` writes gets none, regardless of what else it touched.
    fn also_touched(self, extra: impl IntoIterator<Item = UtteranceIdx>) -> Self {
        match self {
            Self::Suppressed => Self::Suppressed,
            Self::Pending(mut touched) => {
                touched.extend(extra);
                Self::Pending(touched)
            }
        }
    }
}

/// Proof that, at the moment it was built, `utterance` carried at least one
/// word with real timing (main tier or `%wor`, whichever currently carries
/// it -- see [`furthest_word_timing_end`]).
///
/// 2026-09-01 review, item 9. The write phase's own `WorPlan::Pending` list
/// records which utterances a run INJECTED or REFRESHED timing into, but an
/// utterance can lose every one of its words AFTER being added to that list:
/// Pass 1 strips it outright for a start regression, Pass 2's
/// `EndClampedInterleavedWords` can clamp its last word to nothing,
/// `repair_bullets`'s LIS removal strips it, or (the case this type exists
/// to catch, `3009.cha` line 58 in the real Batch 1 fixture) it never had a
/// timed word to begin with: `update_utterance_bullet_with_boundary_policy`'s
/// "no word timings: leave the existing bullet unchanged" branch (`mod.rs`)
/// preserves an INHERITED bullet (from a prior run, or a UTR hint) exactly
/// because there is nothing to derive one from, so the utterance keeps a
/// bullet while every one of its words stays untimed. A `%wor` tier
/// generated from zero timed words is worse than none: it looks like
/// evidence and carries no timing at all.
///
/// So the write phase does not trust `Pending`'s membership; it re-derives
/// this fact FRESH, from whatever state the utterance is actually in at the
/// moment `%wor` would be written, which is after every strip and every
/// clamp above has already run. `write_wor_tier` below takes this proof, not
/// an `UtteranceIdx`, so there is no route to `add_wor_tier` in the write
/// phase that skips the check: an untimed `%wor` tier is unconstructible
/// from this call site, regardless of what `Pending` says.
struct TimedUtterance(UtteranceIdx);

impl TimedUtterance {
    /// The only constructor. `None` when `utterance` currently has no timed
    /// word on any tier.
    fn inspect(utterance_idx: UtteranceIdx, utterance: &Utterance) -> Option<Self> {
        furthest_word_timing_end(utterance).map(|_| Self(utterance_idx))
    }

    fn utterance_idx(self) -> UtteranceIdx {
        self.0
    }
}

/// Proof that `utterance`'s timing was just removed: bullet, main-tier word
/// timings, and `%wor` are all gone.
///
/// Returned rather than left implicit (2026-09-01 review, item 9) so a
/// caller stripping an utterance is handed the fact that any
/// [`TimedUtterance`] proof obtained for it earlier is now stale, instead of
/// having to remember that on its own. No caller of this crate currently
/// needs to ACT on the value (the write phase re-derives timing fresh at
/// write time regardless, which is what actually closes the gap this type
/// documents), so it is not `#[must_use]`; it exists so the signature itself
/// states what stripping does, rather than a caller having to trust a
/// comment.
#[derive(Debug, Clone, Copy)]
pub(super) struct StrippedTiming;

/// Proof that FA results were injected into a file, carrying what that decided.
///
/// # Why the ordering obligation lives HERE
///
/// Monotonicity enforcement must run after injection on the same file. Without
/// it, UTR anchor drift survives into the output, which is how a real delivery
/// shipped backward timestamps. That rule was held by a comment in three
/// places, and the incremental path broke it anyway.
///
/// The obvious cure, making [`enforce_monotonicity`] REQUIRE this type, is
/// WRONG: `fa::run_fa_from_ast` calls it standalone in a pre-FA repair path
/// that has injected nothing, and a precondition would forbid that legitimate
/// use. The constraint is not "enforcement always follows injection", it is
/// "IF you inject, you must then enforce", which is an obligation on the
/// PRODUCER rather than a precondition on the consumer.
///
/// So this type carries the records injection produced and offers no accessor
/// for them. The production transition is [`FaApplied::then_finalize`], which
/// runs optional repair and monotonicity before releasing a state that
/// [`FaDecisions`] accepts. Skipping enforcement now means never obtaining the
/// records structured run evidence needs, which is pressure in the right
/// direction rather than a rule to remember.
#[must_use = "these records must reach structured evidence; call then_finalize"]
pub struct FaApplied {
    postprocess: Vec<batchalign_transform::decisions::DecisionRecord>,
    end_overlap_policy: EndOverlapPolicy,
    /// Which main bullets repair and monotonicity must leave as given. Carried
    /// here, beside the overlap policy, because `then_finalize` is where both
    /// of those phases run and where the given bullets are re-imposed.
    main_bullets: MainBulletAuthority,
    wor_plan: WorPlan,
}

/// What injection and monotonicity enforcement decided, in that order.
///
/// Fields are PRIVATE, and that is what makes [`FaApplied`] a proof rather than
/// a label. They were `pub` for one revision, and the check that was supposed
/// to demonstrate the phase type instead demonstrated the hole: dropping the
/// `FaApplied` and writing `FaOrdered { postprocess: Vec::new(), monotonicity:
/// Vec::new() }` compiled, which is exactly the skipped-enforcement bug the
/// type exists to forbid. The internal `then_enforce_monotonicity` transition
/// is the only constructor, so possession really is the evidence.
struct FaOrdered {
    postprocess: Vec<batchalign_transform::decisions::DecisionRecord>,
    monotonicity: MonotonicityResult,
}

/// Proof that optional repair and monotonicity ran in the sole valid order.
///
/// Keeping repair decisions inside this state prevents full and incremental
/// execution from swapping the phases or omitting repair on a reuse path.
#[must_use = "finalized FA decisions must reach structured evidence"]
pub struct FaFinalized {
    ordered: FaOrdered,
    repair: Vec<batchalign_transform::decisions::DecisionRecord>,
    repair_stats: RepairStats,
    /// Every given bullet, under `--main-bullets keep`, verified unchanged
    /// after every phase; trivially held under the default.
    kept_bullets: KeptBulletsHeld,
}

/// Machine-readable timing change made by monotonicity enforcement.
///
/// The generic decision record remains the human-facing audit message. This
/// enum carries the numeric facts without requiring research code to parse the
/// message string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MonotonicityEffect {
    /// A later utterance started before the greatest preceding start and lost
    /// its timing.
    StartRegressionStripped {
        /// Affected line index, including headers.
        line_idx: usize,
        /// Stable ordinal among utterances only.
        utterance_idx: UtteranceIdx,
        /// Speaker on the affected utterance.
        speaker: String,
        /// Measured start that regressed.
        start_ms: u64,
        /// Greatest preceding start in document order.
        previous_start_ms: u64,
        /// Line that supplied `previous_start_ms`.
        previous_line_idx: usize,
        /// Stable ordinal of the preceding utterance.
        previous_utterance_idx: UtteranceIdx,
        /// Speaker on the line that supplied `previous_start_ms`.
        previous_speaker: String,
    },
    /// Clamping an earlier utterance to the next start would have made it
    /// zero-width, so the earlier timing was removed.
    ZeroDurationClampStripped {
        /// Affected line index, including headers.
        line_idx: usize,
        /// Stable ordinal among utterances only.
        utterance_idx: UtteranceIdx,
        /// Speaker on the affected utterance.
        speaker: String,
        /// Original start of the earlier utterance.
        start_ms: u64,
        /// Original end of the earlier utterance.
        original_end_ms: u64,
        /// Following start that made a positive clamp impossible.
        next_start_ms: u64,
        /// Following line that supplied `next_start_ms`.
        next_line_idx: usize,
        /// Stable ordinal of the following timed utterance.
        next_utterance_idx: UtteranceIdx,
        /// Speaker on the following line.
        next_speaker: String,
    },
    /// A same-speaker end overlap was resolved because only the bullet's
    /// inherited coverage overshot the next utterance's start: no measured
    /// word conflicted. The bullet end moved to the word hull, or (when the
    /// utterance has no measured words at all) to the next start directly.
    /// Words are untouched.
    EndClampedCoverageOnly {
        /// The two utterances involved and how they are identified.
        edge: OverlapEdge,
        /// The end this bullet was clamped to.
        clamped_to_ms: u64,
    },
    /// A same-speaker end overlap was resolved by replacing both
    /// utterances' inherited boundary with their measured word hulls; the
    /// words themselves never conflicted, so the arbitrary next-start clamp
    /// was never applied to either.
    EndClampedBoundaryFromWords {
        /// The two utterances involved and how they are identified.
        edge: OverlapEdge,
        /// The previous utterance's measured last-word end, its new bullet end.
        prev_hull_end_ms: u64,
        /// The next utterance's measured first-word start, its new bullet start.
        next_hull_start_ms: u64,
    },
    /// A same-speaker end overlap could not be resolved from measurement
    /// alone: the words themselves interleave, or the next utterance has
    /// none. The bullet AND every previous-utterance word past the bound
    /// were clamped, and review is requested.
    EndClampedInterleavedWords {
        /// The two utterances involved and how they are identified.
        edge: OverlapEdge,
        /// Following start used as the new end.
        clamped_to_ms: u64,
        /// Words cut to a shorter positive extent, still keeping a timing.
        words_trimmed: usize,
        /// Words whose start was at or past the bound: no timing survived.
        /// One record per word, not just a count (2026-09-02): each is a
        /// MEASURED (or transcript-carried) extent this run threw away, not
        /// the same fact as `words_trimmed`, and a reviewer needs to see
        /// WHICH words and WHAT was lost, not only how many.
        words_dropped: Vec<DroppedWordTiming>,
    },
    /// Under `--main-bullets keep`, a revisable bullet gave way to a kept
    /// neighbour's fixed boundary: the kept bullet's END (the revisable one
    /// followed it and overlapped it) or its START (the revisable one came
    /// before it but started after it). The kept bullet did not change.
    YieldedToKeptBullet {
        /// Affected (yielding) line index, including headers.
        line_idx: usize,
        /// Stable ordinal of the yielding utterance.
        utterance_idx: UtteranceIdx,
        /// Speaker on the yielding utterance.
        speaker: String,
        /// Line of the kept utterance it yielded to.
        kept_line_idx: usize,
        /// Stable ordinal of the kept utterance.
        kept_utterance_idx: UtteranceIdx,
        /// Speaker on the kept utterance.
        kept_speaker: String,
        /// The kept boundary yielded to: the kept end, or the kept start.
        kept_boundary_ms: u64,
        /// What happened to the yielding utterance.
        outcome: KeptBulletYield,
    },
}

/// What a revisable bullet did when it gave way to a kept one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KeptBulletYield {
    /// Its start moved forward to the kept end. Leading words that reached
    /// back before the new start were cut to it (none, when the words already
    /// started after the kept end).
    StartMoved {
        /// Start before the move.
        from_ms: u64,
        /// Start after the move: the kept end.
        to_ms: u64,
        /// Leading words cut to a shorter positive extent.
        words_trimmed: usize,
        /// Leading words with no extent left after the kept end.
        words_dropped: Vec<DroppedWordTiming>,
    },
    /// No valid bullet could remain beside the kept one: timing stripped.
    Stripped {
        /// Start of the stripped bullet.
        start_ms: u64,
        /// End of the stripped bullet.
        end_ms: u64,
    },
}

impl MonotonicityEffect {
    /// The utterance whose WORD timings this effect may have changed, if
    /// any (2026-09-01 review, item 13).
    ///
    /// Only `EndClampedInterleavedWords` clamps a word, and only on its
    /// `edge.utterance_idx` side (the `next` side is never touched under
    /// any resolution). `EndClampedCoverageOnly` and
    /// `EndClampedBoundaryFromWords` move only a BULLET, which `%wor` does
    /// not encode, so they do not make `%wor` stale. The two stripped
    /// variants remove an utterance's timing (and its `%wor`, if any)
    /// outright rather than leaving a mismatch, so there is nothing for the
    /// write phase to regenerate.
    ///
    /// This is the typed refusal item 13 asks for: an utterance this run
    /// never touched, and whose words this pass never mutated, produces NO
    /// `MonotonicityEffect` naming it at all, so it cannot appear here
    /// structurally. `%wor` regeneration can therefore never reach an
    /// utterance a human (or an earlier run) edited and this run left
    /// alone; there is no scan of "every utterance in the file" for such a
    /// value to leak through.
    fn word_mutated_utterance(&self) -> Option<UtteranceIdx> {
        match self {
            Self::EndClampedInterleavedWords { edge, .. } => Some(edge.utterance_idx),
            Self::YieldedToKeptBullet {
                utterance_idx,
                outcome:
                    KeptBulletYield::StartMoved {
                        words_trimmed,
                        words_dropped,
                        ..
                    },
                ..
            } => match *words_trimmed > 0 || !words_dropped.is_empty() {
                true => Some(*utterance_idx),
                false => None,
            },
            Self::YieldedToKeptBullet {
                outcome: KeptBulletYield::Stripped { .. },
                ..
            }
            | Self::StartRegressionStripped { .. }
            | Self::ZeroDurationClampStripped { .. }
            | Self::EndClampedCoverageOnly { .. }
            | Self::EndClampedBoundaryFromWords { .. } => None,
        }
    }
}

/// The two utterances an end-overlap resolution ran on, and how each is
/// identified: line index (for a decision record), stable ordinal, and
/// speaker code.
///
/// Replaces seven fields (`line_idx`, `utterance_idx`, `speaker`,
/// `original_end_ms`, `next_line_idx`, `next_utterance_idx`, `next_speaker`)
/// that were repeated verbatim across three [`MonotonicityEffect`] arms,
/// three [`crate::types::traces::FaTimingDecisionTrace`] arms, the `From`
/// impl between them, and three construction sites in `orchestrate.rs`
/// (2026-09-01 review, item 5). One owner: a caller builds this once, per
/// pair, and every arm embeds it rather than restating its fields.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OverlapEdge {
    /// Affected (previous) line index, including headers.
    pub line_idx: usize,
    /// Stable ordinal of the affected (previous) utterance.
    pub utterance_idx: UtteranceIdx,
    /// Speaker on the affected (previous) utterance.
    pub speaker: String,
    /// The previous utterance's end before clamping.
    pub original_end_ms: u64,
    /// Following line that supplied the clamp boundary.
    pub next_line_idx: usize,
    /// Stable ordinal of the following timed utterance.
    pub next_utterance_idx: UtteranceIdx,
    /// Speaker on the following line.
    pub next_speaker: String,
}

/// The inseparable human-readable and numeric outputs of monotonicity.
#[must_use = "monotonicity decisions must reach review and evidence outputs"]
pub struct MonotonicityResult {
    records: Vec<batchalign_transform::decisions::DecisionRecord>,
    effects: Vec<MonotonicityEffect>,
}

impl MonotonicityResult {
    /// Generic records retained in structured evidence.
    pub fn records(&self) -> &[batchalign_transform::decisions::DecisionRecord] {
        &self.records
    }

    /// Numeric timing effects paired with the generic records.
    pub fn effects(&self) -> &[MonotonicityEffect] {
        &self.effects
    }

    fn into_parts(
        self,
    ) -> (
        Vec<batchalign_transform::decisions::DecisionRecord>,
        Vec<MonotonicityEffect>,
    ) {
        (self.records, self.effects)
    }
}

impl FaApplied {
    /// Fold in utterances a caller separately refreshed from reusable
    /// `%wor` (outside this run's own injection loop) and wants covered by
    /// this SAME write phase, so a caller with a partial-reuse or
    /// incremental path never has an excuse to write `%wor` itself. See
    /// [`WorPlan::Pending`].
    pub fn also_touched(mut self, extra: impl IntoIterator<Item = UtteranceIdx>) -> Self {
        self.wor_plan = self.wor_plan.also_touched(extra);
        self
    }

    /// Impose kept bullets, apply optional repair, enforce the declared
    /// monotonicity policy, write `%wor`, and verify every kept bullet held.
    ///
    /// Repair must see the raw adjacent overlap in order to average a small
    /// boundary. Enforcing monotonicity first would clamp that evidence away
    /// and make `--bullet-repair` behave differently in incremental runs.
    ///
    /// The kept-bullet steps are typed rather than ordered by comment:
    /// repair and monotonicity each REQUIRE the [`ImposedBullets`] proof that
    /// only [`MainBulletAuthority::impose`] returns, and the result carries
    /// the [`KeptBulletsHeld`] proof [`MainBulletAuthority::verify_held`]
    /// returns after the last phase. Under the default policy both are
    /// trivially obtained and nothing changes. A kept bullet that did not
    /// hold fails the file with a [`KeptBulletError`].
    pub fn then_finalize(
        self,
        chat_file: &mut ChatFile,
        repair_policy: BulletRepairPolicy,
    ) -> Result<FaFinalized, KeptBulletError> {
        let FaApplied {
            mut postprocess,
            end_overlap_policy,
            main_bullets,
            wor_plan,
        } = self;
        let (imposed, cuts, kept_records) = main_bullets.impose(chat_file)?.into_parts();
        postprocess.extend(kept_records);
        // A kept utterance whose main-tier words were cut gets its `%wor`
        // regenerated from them when a write is requested; under
        // `Suppressed`, `impose` already clamped the existing `%wor` in place.
        let wor_plan = wor_plan.also_touched(cuts.into_utterances());
        let repair_result = match repair_policy {
            BulletRepairPolicy::Disabled => Default::default(),
            BulletRepairPolicy::Enabled => {
                repair_bullets_under(chat_file, false, end_overlap_policy, imposed)?
            }
        };
        let repair = repair_result.decisions.iter().map(Into::into).collect();
        let ordered = enforce_monotonicity_then_write_wor(
            chat_file,
            end_overlap_policy,
            imposed,
            wor_plan,
            postprocess,
        )?;
        let kept_bullets = main_bullets.verify_held(chat_file)?;
        Ok(FaFinalized {
            ordered,
            repair,
            repair_stats: repair_result.stats,
            kept_bullets,
        })
    }
}

/// Enforce monotonicity, then (re)write `%wor` per `wor_plan`, and hand back
/// both sets of records.
///
/// Private and reached only from [`FaApplied::then_finalize`], which is what
/// makes the sequence unskippable: there is no other way to reach the
/// injection records. The `%wor` write happens HERE, after monotonicity has
/// resolved every same-speaker overlap, which is the whole point of carrying
/// `wor_plan` instead of writing `%wor` inline as each utterance was
/// processed: this is the only place in the type graph that can produce a
/// `FaOrdered`, so it is the only place a caller of this crate can cause a
/// `%wor` tier to be written, and it always happens after resolution.
fn enforce_monotonicity_then_write_wor(
    chat_file: &mut ChatFile,
    end_overlap_policy: EndOverlapPolicy,
    imposed: ImposedBullets<'_>,
    wor_plan: WorPlan,
    postprocess: Vec<batchalign_transform::decisions::DecisionRecord>,
) -> Result<FaOrdered, KeptBulletError> {
    let monotonicity = enforce_monotonicity_under(chat_file, end_overlap_policy, imposed)?;
    // Fold in whichever utterances THIS resolution itself clamped a word
    // on, even when the run never touched them by injection or refresh
    // (2026-09-01 review, item 13): an `EndClampedInterleavedWords` can
    // land on either side of a pair where only ONE side was `touched`.
    // `also_touched` is a no-op when `write_wor` was never requested, so
    // this cannot cause a write that was not asked for.
    let wor_plan = wor_plan.also_touched(
        monotonicity
            .effects()
            .iter()
            .filter_map(MonotonicityEffect::word_mutated_utterance),
    );
    if let WorPlan::Pending(touched) = wor_plan {
        // A `BTreeSet`: `also_touched` can name the same utterance twice
        // (originally touched AND word-mutated), and regenerating `%wor`
        // twice for one utterance is wasted work, not a correctness
        // question, but there is no reason to do it.
        let candidates: std::collections::BTreeSet<UtteranceIdx> = touched.into_iter().collect();
        for utt_idx in candidates {
            // The write phase does not trust its own candidate list:
            // `TimedUtterance::inspect` re-derives, from the utterance's
            // CURRENT words, whether there is anything to write at all
            // (2026-09-01 review, item 9). An untimed `%wor` tier is
            // unconstructible from this call site.
            let timed = get_utterance_mut(chat_file, utt_idx)
                .and_then(|utt| TimedUtterance::inspect(utt_idx, &*utt));
            if let Some(timed) = timed
                && let Some(utt) = get_utterance_mut(chat_file, timed.utterance_idx())
            {
                add_wor_tier(utt);
            }
        }
    }
    Ok(FaOrdered {
        postprocess,
        monotonicity,
    })
}

impl FaOrdered {
    /// Monotonicity effects proved to have run in this ordered phase.
    pub fn monotonicity(&self) -> &MonotonicityResult {
        &self.monotonicity
    }
}

impl FaFinalized {
    /// Monotonicity effects proved to have run after optional repair.
    pub fn monotonicity(&self) -> &MonotonicityResult {
        self.ordered.monotonicity()
    }

    /// Aggregate observations from the optional repair phase.
    pub fn repair_stats(&self) -> &RepairStats {
        &self.repair_stats
    }

    /// The proof that every kept bullet held.
    pub fn kept_bullets(&self) -> KeptBulletsHeld {
        self.kept_bullets
    }
}

fn projection_without_injection(projection: FaProjection) -> FaApplied {
    let (policy, main_bullets) = projection.into_parts();
    FaApplied {
        postprocess: Vec::new(),
        end_overlap_policy: policy.end_overlaps(),
        main_bullets,
        wor_plan: WorPlan::Suppressed,
    }
}

/// A no-injection projection that still has a known set of utterances a
/// caller separately refreshed from reusable `%wor` and wants covered by
/// this run's write phase (2026-09-01 review, item 2): the "all utterances
/// reusable" fast path refreshes every utterance's timing before this
/// projection ever runs, and that refresh must not write `%wor` itself.
pub fn projection_without_injection_with_touched(
    projection: FaProjection,
    write_wor: bool,
    touched: Vec<UtteranceIdx>,
) -> FaApplied {
    let (policy, main_bullets) = projection.into_parts();
    FaApplied {
        postprocess: Vec::new(),
        end_overlap_policy: policy.end_overlaps(),
        main_bullets,
        wor_plan: WorPlan::new(write_wor, touched),
    }
}

/// Finalize a no-injection projection through the same repair/order typestate.
pub fn finalize_without_injection(
    chat_file: &mut ChatFile,
    projection: FaProjection,
    repair_policy: BulletRepairPolicy,
) -> Result<FaFinalized, KeptBulletError> {
    projection_without_injection(projection).then_finalize(chat_file, repair_policy)
}

/// Apply FA results to a `ChatFile`: inject timings, postprocess, and optionally
/// generate `%wor` before the required finalization transition.
///
/// `groups` and `responses` must be parallel: `responses[i]` is the aligned
/// timings for `groups[i]`.
///
/// When `write_wor` is `true`, a `%wor` tier is generated for each utterance.
/// When `false`, existing `%wor` tiers are left untouched and no new ones are added.
///
/// Returns [`FaApplied`], which is the only route to the records this made and
/// which cannot be read without running monotonicity enforcement. See that
/// type for why the obligation lives there rather than on this signature.
///
/// `#[cfg(test)]`: derive-only, with no production caller. Production goes
/// through [`apply_fa_results_with_projection_policy`] with a bound
/// [`FaProjection`], so this shortcut cannot route around `--main-bullets`.
#[cfg(test)]
pub fn apply_fa_results(
    chat_file: &mut ChatFile,
    groups: &[FaGroup],
    responses: &[Vec<Option<WordTiming>>],
    policy: WordEndPolicy,
    write_wor: bool,
) -> FaApplied {
    apply_fa_results_with_projection_policy(
        chat_file,
        groups,
        responses,
        FaProjection::new(
            super::FaProjectionPolicy::new(
                policy,
                ExistingWorBoundaryPolicy::Preserve,
                EndOverlapPolicy::ClampAllAdjacent,
            ),
            MainBulletAuthority::DeriveFromWords,
        ),
        write_wor,
    )
    .expect("the default policy refers to no given bullet")
}

/// Apply FA evidence under an explicit, complete CHAT projection policy.
///
/// This is the production entry point. [`apply_fa_results`] remains as the
/// compatibility helper whose old signature necessarily selects the historic
/// prior-boundary behavior.
///
/// Under `--main-bullets keep`, an utterance whose bullet the input gave is
/// never passed to the bullet update; `MainBulletAuthority::impose` puts the
/// given bullet back and fits the words at finalization. An utterance the
/// input left unbulleted derives its bullet exactly as it does by default.
pub fn apply_fa_results_with_projection_policy(
    chat_file: &mut ChatFile,
    groups: &[FaGroup],
    responses: &[Vec<Option<WordTiming>>],
    projection: FaProjection,
    write_wor: bool,
) -> Result<FaApplied, KeptBulletError> {
    let (policy, main_bullets) = projection.into_parts();
    let mut decisions = Vec::new();
    let mut touched: Vec<UtteranceIdx> = Vec::new();
    // 0. Strip stale decision tiers (%xalign / %xrev) from any previous FA run.
    //
    // This must happen unconditionally, not gated on whether new decisions will
    // be produced.  Without an unconditional strip, a clean re-run (no new
    // decisions) leaves the previous run's tiers in place; the NEXT run that
    // DOES produce decisions then appends to them, creating duplicates.
    batchalign_transform::decisions::strip_decision_tiers(chat_file);

    // 1. Strip InternalBullet tokens left over from parsing.
    //
    // When the FA pipeline receives CHAT text that already has UTR-injected
    // bullets, the parser creates InternalBullet content items. FA then
    // sets word inline_bullet + utterance .bullet via update_utterance_bullet.
    // Without stripping, both the old InternalBullet AND the new .bullet
    // are serialized, producing "two timestamps on the main line".
    for (group, _) in groups.iter().zip(responses.iter()) {
        for &utt_idx in &group.utterance_indices {
            if let Some(utt) = get_utterance_mut(chat_file, utt_idx) {
                strip_internal_bullet_tokens(&mut utt.main.content.content);
            }
        }
    }

    // 2. Inject each utterance's timings and post-process it, in ONE pass.
    //
    // Injection and post-processing were two loops over the same index
    // sequence. Writing a timing into a `Bullet` DESTROYS its provenance (a
    // bullet is two integers), and the second loop used to recover the spans by
    // reading those bullets back, which relabelled every invented timing as
    // something the transcript had always carried, and therefore as an
    // observation. Carrying the injected values forward is the only way
    // post-processing can know what it is adjusting.
    //
    // Fusing the loops keeps that evidence a LOCAL. Passing it between two
    // loops needed a `HashMap<UtteranceIdx, InjectedTimings>` that was pure
    // indirection: the second loop walked `groups.flat_map(utterance_indices)`,
    // exactly what the first walked, and `collect_final_timings` refuses unless
    // `responses.len() == groups.len()`, so every lookup hit. On a
    // 15,000-utterance transcript that map held about 5.8 MB across 15,000
    // allocations, and it made a "this utterance has no evidence" case
    // reachable in the types that could not occur in practice.
    for (group, timings) in groups.iter().zip(responses.iter()) {
        let mut timing_offset: usize = 0;

        for &utt_idx in &group.utterance_indices {
            // Resolved before the mutable borrow. `utt_idx` is an utterance
            // ordinal and `DecisionRecord.line_idx` indexes lines; they never
            // coincide, because every CHAT file opens with headers. This used
            // to pass the ordinal straight through, which silently dropped the
            // decision (the consumer found a header at that index and skipped
            // it) or attached it to the wrong utterance.
            let line_idx = super::utterance_line_idx(chat_file, utt_idx);
            let Some(utt) = get_utterance_mut(chat_file, utt_idx) else {
                continue;
            };

            let mutability = main_bullets.given_mutability(utt_idx)?;

            let injected = inject_timings_for_utterance(utt, timings, &mut timing_offset);
            let outcome = postprocess_utterance_timings_with_boundary_policy(
                utt,
                policy.word_ends(),
                policy.existing_wor_boundaries(),
                &injected,
            );

            if let (true, Some(line_idx)) = (outcome.dropped.any(), line_idx) {
                decisions.push(
                    batchalign_transform::decisions::DecisionRecord::new_and_trace(
                        line_idx.raw(),
                        utt.main.speaker.as_str().to_string(),
                        batchalign_transform::decisions::DecisionStrategy::Fa(
                            batchalign_transform::decisions::FaStrategy::WordsTimingDropped,
                        ),
                        outcome.dropped.reason(),
                        true,
                    ),
                );
            }
            // The provenance summary is the only form in which per-word origins
            // can cross into the transcript, since the write above lowers them
            // into `Bullet`. Emitted only when something was not a
            // straightforward observation, so the tier means something when it
            // appears.
            if let (true, Some(line_idx)) = (outcome.provenance.any_not_observed(), line_idx) {
                decisions.push(
                    batchalign_transform::decisions::DecisionRecord::new_and_trace(
                        line_idx.raw(),
                        utt.main.speaker.as_str().to_string(),
                        batchalign_transform::decisions::DecisionStrategy::Fa(
                            batchalign_transform::decisions::FaStrategy::TimingProvenance,
                        ),
                        format!("{}", outcome.provenance),
                        outcome.provenance.needs_review(),
                    ),
                );
            }

            match mutability {
                // Read-only: `MainBulletAuthority::impose` restores the given
                // bullet and cuts the words to fit it at finalization, or,
                // for a kept absence, leaves it without one and untimes its
                // words.
                GivenMutability::ReadOnly => {}
                GivenMutability::Revisable => update_utterance_bullet_with_boundary_policy(
                    utt,
                    policy.existing_wor_boundaries(),
                ),
            }
            touched.push(utt_idx);
        }
    }

    // NOTE: E362 (monotonicity) and E704 (same-speaker overlap) enforcement
    // was removed here. These passes aggressively stripped timing from
    // utterances that had "imperfect but usable" timings, causing severe
    // regressions vs batchalign 0.8.x (up to 60% timing loss on real data).
    // The CHAT validator in talkbank-tools flags these violations after the
    // fact: the FA pipeline should not silently destroy timing data.
    //
    // `%wor` is deliberately NOT written here (it used to be, per
    // utterance, right above). See `WorPlan`'s doc comment: writing it
    // before monotonicity resolves same-speaker overlaps can leave it stale
    // relative to a word timing monotonicity later cuts.
    // `FaApplied::then_enforce_monotonicity` writes it, once, after.

    Ok(FaApplied {
        postprocess: decisions,
        end_overlap_policy: policy.end_overlaps(),
        main_bullets,
        wor_plan: WorPlan::new(write_wor, touched),
    })
}

/// Refresh a CHAT file that already carries reusable `%wor` timing.
///
/// This is the cheap rerun path for `align`. Instead of sending audio back
/// through the FA worker, the function:
///
/// 1. aligns each `%wor` tier back to the main tier,
/// 2. rehydrates main-tier `inline_bullet` timing from `%wor`,
/// 3. removes any parsed `InternalBullet` tokens left over from roundtripped
///    serialization,
/// 4. recomputes utterance bullets, and
/// 5. optionally regenerates `%wor`.
///
/// Callers should only use this after [`super::has_reusable_wor_timing`]
/// succeeds.
///
/// `#[cfg(test)]` (2026-09-01 review, item 12): it has no production caller
/// (see [`refresh_existing_alignment_with_boundary_policy`]'s own doc) and
/// writes `%wor` directly, bypassing the write phase; gating it to test
/// builds is what makes [`add_wor_tier`]'s doc claim ("reachable only from
/// the write phase") actually true outside test code, rather than true only
/// by nobody happening to call this in production.
#[cfg(test)]
pub fn refresh_existing_alignment(chat_file: &mut ChatFile, write_wor: bool) {
    refresh_existing_alignment_with_boundary_policy(
        chat_file,
        write_wor,
        ExistingWorBoundaryPolicy::Preserve,
    );
}

/// Refresh every reusable `%wor` tier under an explicit boundary policy,
/// optionally writing `%wor` from the refreshed state directly.
///
/// This whole-file convenience wrapper has NO production caller
/// (`fa::run_fa_from_ast`'s cheap-rerun path, the one that used to call it,
/// now builds its own `FaApplied` via `refresh_reusable_alignment` +
/// [`projection_without_injection_with_touched`] instead, so it can retain
/// the monotonicity records and run bullet repair; see that call site). It
/// remains, alongside [`add_wor_tier`], as a test-fixture convenience for
/// building a file with `%wor` already present and CURRENT (not yet
/// monotonicity-resolved), which several tests deliberately need as the
/// STARTING point for exercising a separate monotonicity/finalize step.
/// Because it has no production caller, writing `%wor` here without running
/// monotonicity does not reopen the ordering defect this file's `WorPlan`
/// exists to close (2026-09-01 review, item 2): nothing in the real pipeline
/// reaches this function any more.
///
/// `#[cfg(test)]` (2026-09-01 review, item 12), for the same reason: this
/// crate no longer builds a `%wor`-writing binary that can reach this
/// function, so nothing but a test can construct the call that reaches
/// [`add_wor_tier`] through it.
#[cfg(test)]
pub fn refresh_existing_alignment_with_boundary_policy(
    chat_file: &mut ChatFile,
    write_wor: bool,
    existing_wor_boundaries: ExistingWorBoundaryPolicy,
) {
    for utt_idx in refresh_reusable_alignment(chat_file, existing_wor_boundaries) {
        if write_wor && let Some(utt) = get_utterance_mut(chat_file, utt_idx) {
            add_wor_tier(utt);
        }
    }
}

/// The mechanical part of [`refresh_existing_alignment_with_boundary_policy`]:
/// rehydrate every reusable utterance's timing, WITHOUT writing `%wor`.
/// Returns the utterances actually refreshed, for a caller to fold into
/// whichever `%wor` write phase it is already running (see
/// [`projection_without_injection_with_touched`] / [`FaApplied::also_touched`]).
pub fn refresh_reusable_alignment(
    chat_file: &mut ChatFile,
    existing_wor_boundaries: ExistingWorBoundaryPolicy,
) -> Vec<UtteranceIdx> {
    let mut touched = Vec::new();
    let mut utt_idx = 0usize;
    for line in &mut chat_file.lines {
        let Line::Utterance(utterance) = line else {
            continue;
        };
        let utterance_idx = UtteranceIdx::new(utt_idx);
        utt_idx += 1;
        if count_alignable_main_words(utterance) == 0 {
            continue;
        }

        let refreshed = refresh_existing_alignment_for_utterance_with_boundary_policy(
            utterance,
            existing_wor_boundaries,
        );
        if refreshed {
            touched.push(utterance_idx);
        } else {
            tracing::warn!(
                "skipping utterance with unreusable %wor timing in refresh_existing_alignment"
            );
        }
    }
    touched
}

/// Return `true` when one utterance has reusable `%wor` timing.
///
/// This is the per-utterance form of the cheap rerun check. It is useful for
/// selective reuse in incremental align workflows where only some utterances
/// remain trustworthy after manual edits.
pub fn has_reusable_wor_timing_for_utterance(utterance: &Utterance) -> bool {
    collect_wor_backed_timings(utterance).is_some()
}

/// Refresh one utterance from its existing `%wor` timing. Mechanical only:
/// never writes `%wor` itself (see [`add_wor_tier`]'s doc for the routes that
/// do). Returns `true` when the utterance had a clean reusable `%wor`
/// mapping and was refreshed successfully; `false` when `%wor` was missing,
/// mismatched, or partially untimed.
pub fn refresh_existing_alignment_for_utterance(utterance: &mut Utterance) -> bool {
    refresh_existing_alignment_for_utterance_with_boundary_policy(
        utterance,
        ExistingWorBoundaryPolicy::Preserve,
    )
}

/// Refresh one reusable `%wor` tier under an explicit boundary policy.
/// Mechanical only: see [`refresh_existing_alignment_for_utterance`].
fn refresh_existing_alignment_for_utterance_with_boundary_policy(
    utterance: &mut Utterance,
    existing_wor_boundaries: ExistingWorBoundaryPolicy,
) -> bool {
    let Some(timings) = collect_wor_backed_timings(utterance) else {
        return false;
    };

    strip_internal_bullet_tokens(&mut utterance.main.content.content);
    let mut offset = 0usize;
    inject_timings_for_utterance(utterance, &timings, &mut offset);
    update_utterance_bullet_with_boundary_policy(utterance, existing_wor_boundaries);
    true
}

/// Refresh timing for utterances with reusable `%wor`, leaving stale ones
/// untouched for FA worker processing. Mechanical only, like
/// [`refresh_existing_alignment_for_utterance`]: never writes `%wor` itself.
/// Returns the utterances actually refreshed, for a caller to fold into
/// whichever `%wor` write phase it is already running (see
/// [`FaApplied::also_touched`]).
///
/// This is the per-utterance counterpart to [`refresh_existing_alignment()`].
/// Unlike that function (which asserts all utterances are reusable), this one
/// only refreshes utterances in the provided set, skipping stale ones that
/// will go through FA workers.
pub fn refresh_reusable_utterances(
    chat_file: &mut ChatFile,
    reusable_indices: &std::collections::HashSet<usize>,
) -> Vec<UtteranceIdx> {
    let mut touched = Vec::new();
    let mut utt_idx = 0usize;
    for line in &mut chat_file.lines {
        let Line::Utterance(utterance) = line else {
            continue;
        };
        if reusable_indices.contains(&utt_idx) {
            let refreshed = refresh_existing_alignment_for_utterance(utterance);
            debug_assert!(
                refreshed,
                "utterance {utt_idx} was in reusable set but refresh failed"
            );
            touched.push(UtteranceIdx::new(utt_idx));
        }
        utt_idx += 1;
    }
    touched
}

pub(super) fn furthest_word_timing_end(utterance: &Utterance) -> Option<u64> {
    // Fresh FA evidence lives on main-tier words before `%wor` generation and
    // remains there when the user disables `%wor` output. Review policy must
    // not become weaker merely because that serialization choice is off.
    let mut main_end = None;
    walk_words(&utterance.main.content.content, None, &mut |item| {
        let word = match item {
            WordItem::Word(word) => word,
            WordItem::ReplacedWord(replaced) => &replaced.word,
            WordItem::Separator(_) => return,
        };
        if let Some(end_ms) = word
            .inline_bullet
            .as_ref()
            .map(|bullet| bullet.timing.end_ms)
        {
            main_end = Some(main_end.map_or(end_ms, |current: u64| current.max(end_ms)));
        }
    });

    // Parsed legacy/rerun input may still carry its durable word evidence only
    // on `%wor`, before the refresh path rehydrates main-tier inline bullets.
    let wor_end = utterance.wor_tier().and_then(|wor| {
        wor.items
            .iter()
            .filter_map(|item| match item {
                WorItem::Word(word) => word
                    .inline_bullet
                    .as_ref()
                    .map(|bullet| bullet.timing.end_ms),
                WorItem::Separator { .. } => None,
            })
            .max()
    });

    main_end.into_iter().chain(wor_end).max()
}

/// The earliest measured word start in `utterance`, mirroring
/// [`furthest_word_timing_end`] for the opposite boundary. Checks both the
/// main tier's inline word bullets and a `%wor` tier's, since either
/// representation may be the one currently carrying durable per-word timing
/// (see [`furthest_word_timing_end`]'s own comment for why both exist).
pub(super) fn earliest_word_timing_start(utterance: &Utterance) -> Option<u64> {
    let mut main_start = None;
    walk_words(&utterance.main.content.content, None, &mut |item| {
        let word = match item {
            WordItem::Word(word) => word,
            WordItem::ReplacedWord(replaced) => &replaced.word,
            WordItem::Separator(_) => return,
        };
        if let Some(start_ms) = word
            .inline_bullet
            .as_ref()
            .map(|bullet| bullet.timing.start_ms)
        {
            main_start = Some(main_start.map_or(start_ms, |current: u64| current.min(start_ms)));
        }
    });

    let wor_start = utterance.wor_tier().and_then(|wor| {
        wor.items
            .iter()
            .filter_map(|item| match item {
                WorItem::Word(word) => word
                    .inline_bullet
                    .as_ref()
                    .map(|bullet| bullet.timing.start_ms),
                WorItem::Separator { .. } => None,
            })
            .min()
    });

    main_start.into_iter().chain(wor_start).min()
}

/// The `%wor` tier, mutable, if `utterance` has one.
///
/// `Utterance` exposes `wor_tier()` (immutable) and tiers like `%mor` have a
/// `_mut` companion; `%wor` does not, so this mirrors that pattern locally
/// rather than reading the tier back out through text.
fn wor_tier_mut(utterance: &mut Utterance) -> Option<&mut WorTier> {
    utterance
        .dependent_tiers
        .iter_mut()
        .find_map(|entry| match &mut entry.tier {
            DependentTier::Wor(tier) => Some(tier),
            _ => None,
        })
}

/// Which tier a dropped word's timing lived on.
///
/// Main-tier inline bullets and `%wor` are walked, and indexed, completely
/// independently (a re-run transcript may carry durable timing only on
/// `%wor`), each numbering starting at 0. A bare `word_index` alone cannot
/// say which numbering it counts against, so [`DroppedWordTiming`] carries
/// this alongside it rather than leaving the tier to be inferred from
/// context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WordTier {
    /// The main tier's own inline bullet.
    MainTier,
    /// The `%wor` tier.
    Wor,
}

/// One word whose timing was thrown away entirely, because its start
/// already sat at or past the clamp bound and
/// [`super::postprocess::clamp_transcript_word_to_bullet`] could not trim it
/// to a positive extent.
///
/// This is the caller-facing half of
/// [`super::postprocess::WordClampOutcome::DroppedPastBound`]: that variant
/// already carries the lost span at the point the loss is decided, but the
/// span used to die one layer up, at `clamp_words_past_bound`, which folded
/// every drop into a bare `usize`. Carrying the record itself instead of a
/// count is practice 15 (a return type too weak to hold what was learned)
/// applied at the NEXT boundary the same fact crosses, not a new instance of
/// the defect: a `usize` still answers "how many", `.len()` on the `Vec`
/// this type populates, but only this carries "which words, and what they
/// measured".
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DroppedWordTiming {
    /// This word's position among the words visited on its OWN tier, in
    /// document order (see [`WordTier`]: not a single index shared across
    /// both tiers).
    pub word_index: usize,
    /// Which tier this word's dropped timing lived on.
    pub tier: WordTier,
    /// The extent that was lost.
    pub measured: TimeSpan,
}

/// How many words a clamp cut, split by what actually happened to each and,
/// for trims, by tier.
///
/// The former return type (`usize`) conflated two different facts: a word
/// TRIMMED to a shorter positive extent, which still carries its own new
/// timing in the AST, so a count is all a caller needs; and a word DROPPED
/// because nothing inside the bound survived, which is a MEASURED timing this
/// pass threw away with nothing else left to reconstruct it from. `dropped`
/// is therefore a `Vec`, not a second `usize`: a caller that only needs
/// "how many" reads `.len()`, and nothing keeps a separately-incremented
/// count that could drift from the list it is supposedly counting (shape D).
///
/// Trims are counted per tier because the `%wor` write phase needs to know
/// whether the MAIN tier changed (it regenerates `%wor` from the main tier);
/// a drop already carries its tier in [`DroppedWordTiming`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct ClampedWordCounts {
    /// Main-tier words cut to a shorter positive extent.
    trimmed_main_tier: usize,
    /// `%wor` words cut to a shorter positive extent.
    trimmed_wor: usize,
    /// Words with no positive extent inside the bound. One record per word.
    pub dropped: Vec<DroppedWordTiming>,
}

impl ClampedWordCounts {
    /// Words cut to a shorter positive extent, on either tier. A count only:
    /// the AST itself still holds each one's new timing.
    pub(super) fn trimmed(&self) -> usize {
        self.trimmed_main_tier + self.trimmed_wor
    }

    /// Whether any main-tier word was trimmed or dropped.
    pub(super) fn cut_main_tier(&self) -> bool {
        self.trimmed_main_tier > 0
            || self
                .dropped
                .iter()
                .any(|dropped| dropped.tier == WordTier::MainTier)
    }

    /// Record what happened to one word, returning the inline bullet it
    /// should now carry (`None` when its timing was dropped).
    fn record(
        &mut self,
        word_index: usize,
        tier: WordTier,
        outcome: super::postprocess::WordClampOutcome,
    ) -> Option<Bullet> {
        match outcome {
            super::postprocess::WordClampOutcome::Trimmed(timing) => {
                match tier {
                    WordTier::MainTier => self.trimmed_main_tier += 1,
                    WordTier::Wor => self.trimmed_wor += 1,
                }
                Some(Bullet::new(timing.start_ms, timing.end_ms))
            }
            super::postprocess::WordClampOutcome::DroppedPastBound { measured } => {
                self.dropped.push(DroppedWordTiming {
                    word_index,
                    tier,
                    measured,
                });
                None
            }
        }
    }

    /// Clamp one word into `bound`, when it is timed and reaches outside it.
    fn clamp_word(&mut self, word: &mut Word, word_index: usize, tier: WordTier, bound: TimeSpan) {
        let Some(span) = super::get_word_timing(word) else {
            return;
        };
        if span.start_ms >= bound.start_ms && span.end_ms <= bound.end_ms {
            return;
        }
        let outcome = super::postprocess::clamp_transcript_word_to_bullet(
            span,
            span.start_ms.max(bound.start_ms),
            span.end_ms.min(bound.end_ms),
        );
        word.inline_bullet = self.record(word_index, tier, outcome);
    }
}

/// Clamp every word timing in `utterance` that ends after `bound_ms`. The
/// one-sided case of [`clamp_words_within`], kept as a name because the
/// overlap resolutions only ever cut a word's END.
pub(super) fn clamp_words_past_bound(
    utterance: &mut Utterance,
    bound_ms: u64,
) -> ClampedWordCounts {
    clamp_words_within(utterance, TimeSpan::new(0, bound_ms))
}

/// Clamp every word timing in `utterance` into `bound`, on whichever
/// representation currently carries real per-word timing (main-tier inline
/// bullets, a `%wor` tier, or both; a word can be cut on one, the other, or
/// both independently, since a reparsed transcript may carry durable timing
/// only on `%wor`). A word crossing an edge is cut to it; a word with no
/// positive extent left inside the bound (wholly outside it, or `bound` is
/// empty) is dropped rather than written with no extent. Every cut goes
/// through [`super::postprocess::clamp_transcript_word_to_bullet`], the one
/// route by which a pass after post-processing may shorten a word.
///
/// Returns what was cut, split by [`ClampedWordCounts`], for the caller's
/// decision record and for the `%wor` write phase.
pub(super) fn clamp_words_within(utterance: &mut Utterance, bound: TimeSpan) -> ClampedWordCounts {
    let mut clamped = ClampedWordCounts::default();

    let mut main_word_idx = 0usize;
    walk_words_mut(
        utterance.main.content.content.as_mut_slice(),
        None,
        &mut |leaf| {
            let word = match leaf {
                WordItemMut::Word(word) => word,
                WordItemMut::ReplacedWord(replaced) => &mut replaced.word,
                WordItemMut::Separator(_) => return,
            };
            clamped.clamp_word(word, main_word_idx, WordTier::MainTier, bound);
            main_word_idx += 1;
        },
    );

    if let Some(wor) = wor_tier_mut(utterance) {
        let words = wor.items.iter_mut().filter_map(|item| match item {
            WorItem::Word(word) => Some(word),
            WorItem::Separator { .. } => None,
        });
        for (wor_word_idx, word) in words.enumerate() {
            clamped.clamp_word(word, wor_word_idx, WordTier::Wor, bound);
        }
    }

    clamped
}

/// How a same-speaker end overlap between two adjacent utterances resolves,
/// decided from what is MEASURED (word timings), never from the
/// coverage-extended bullet alone.
///
/// The bullet each utterance carries at this point is a projection: the word
/// hull unioned with whatever legacy coverage an earlier pass preserved
/// (`update_utterance_bullet`'s `Preserve` policy never shrinks a bullet).
/// That union is exactly what can push an END past the next utterance's
/// START even when the words themselves never conflict, which is the defect
/// this type exists to make unrepresentable: these three cases are the only
/// ones the measured hulls can produce, and each is resolved differently.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum EndOverlapResolution {
    /// The previous utterance has no measured word past the next
    /// utterance's start: either it has no timed words at all, or its last
    /// one already ends at or before that start. Only the bullet's
    /// inherited coverage overshot; the bullet end moves to the word hull
    /// (or, with no words at all, to the next start directly, matching the
    /// prior bookkeeping-only clamp). Words are untouched either way.
    CoverageOnly {
        /// The previous utterance's last measured word end, when it has one.
        hull_end_ms: Option<u64>,
    },
    /// The previous utterance's last measured word ends after the next
    /// start, but the two utterances' words do not interleave: the next
    /// utterance's first measured word starts at or after that end. The
    /// boundary is itself measurable, so both bullets take their word-hull
    /// edges instead of the arbitrary next-start clamp; the far side of
    /// each bullet (the previous utterance's start, the next utterance's
    /// end) is untouched.
    BoundaryFromWords {
        /// The previous utterance's measured last-word end.
        prev_hull_end_ms: u64,
        /// The next utterance's measured first-word start.
        next_hull_start_ms: u64,
    },
    /// The two utterances' measured words overlap in time, or the next
    /// utterance has no measured word to fix a boundary against. A genuine
    /// conflict between segmentation and FA: the bullet is clamped to the
    /// next start and every previous-utterance word past that bound is
    /// clamped with it, flagged for human review.
    InterleavedWords,
}

/// Classify a same-speaker end overlap from measured word timings.
///
/// `prev_hull_end_ms` and `next_hull_start_ms` are the previous utterance's
/// last measured word end and the next utterance's first measured word
/// start (each `None` when that utterance has no measured word), and
/// `next_start_ms` is the bullet-level boundary that triggered the pass.
pub(super) fn classify_end_overlap(
    prev_hull_end_ms: Option<u64>,
    next_start_ms: u64,
    next_hull_start_ms: Option<u64>,
    next_successor_start_ms: Option<u64>,
) -> EndOverlapResolution {
    match prev_hull_end_ms {
        None => EndOverlapResolution::CoverageOnly { hull_end_ms: None },
        Some(hull_end_ms) if hull_end_ms <= next_start_ms => EndOverlapResolution::CoverageOnly {
            hull_end_ms: Some(hull_end_ms),
        },
        Some(hull_end_ms) => match next_hull_start_ms {
            // `BoundaryFromWords` moves `next`'s bullet start forward to its
            // measured hull start. Left unguarded, that move can push it past
            // `next`'s OWN successor's start, a fresh non-monotonic-start
            // conflict pass 1 (which already ran) cannot see. `next_successor_start_ms`
            // is that successor's LIVE start, read by the caller before the
            // move; when it would be violated this is a genuine three-utterance
            // conflict, not a clean two-utterance boundary, so it falls back to
            // `InterleavedWords`: `next`'s start is left untouched, and `prev`
            // is clamped (bullet and words) to `next`'s ORIGINAL start instead.
            Some(next_hull_start_ms)
                if next_hull_start_ms >= hull_end_ms
                    && match next_successor_start_ms {
                        Some(successor_start_ms) => next_hull_start_ms < successor_start_ms,
                        None => true,
                    } =>
            {
                EndOverlapResolution::BoundaryFromWords {
                    prev_hull_end_ms: hull_end_ms,
                    next_hull_start_ms,
                }
            }
            _ => EndOverlapResolution::InterleavedWords,
        },
    }
}

impl EndOverlapResolution {
    /// The same resolution, when the NEXT utterance's bullet is kept as given.
    ///
    /// `BoundaryFromWords` is the only resolution that moves the next
    /// utterance's start. With that start fixed, the overlap is resolved the
    /// way `classify_end_overlap` already resolves a pair whose next start may
    /// not move (its successor guard): `InterleavedWords`, which clamps only
    /// the previous utterance, bullet and words, to the next start.
    fn keeping_next_start(self) -> Self {
        match self {
            Self::CoverageOnly { .. } | Self::InterleavedWords => self,
            Self::BoundaryFromWords { .. } => Self::InterleavedWords,
        }
    }

    /// Whether this resolution needs human review: only a genuine word
    /// conflict does. Derived from the variant, not carried beside it (see
    /// this type's own doc for why a former `cuts_word_timing` boolean was
    /// exactly the affordance a type here replaces).
    fn needs_review(&self) -> bool {
        matches!(self, Self::InterleavedWords)
    }

    /// The wire-label strategy this resolution corresponds to (2026-09-01
    /// review, item 4): `MonotonicityStrategy`'s three `EndClamped*`
    /// variants exist one-for-one with this type's, so a consumer filtering
    /// on the wire label sees the same three cases this type does.
    fn strategy(&self) -> batchalign_transform::decisions::MonotonicityStrategy {
        use batchalign_transform::decisions::MonotonicityStrategy;
        match self {
            Self::CoverageOnly { .. } => MonotonicityStrategy::EndClampedCoverageOnly,
            Self::BoundaryFromWords { .. } => MonotonicityStrategy::EndClampedBoundaryFromWords,
            Self::InterleavedWords => MonotonicityStrategy::EndClampedInterleavedWords,
        }
    }

    /// The human-facing decision reason, derived from the variant plus the
    /// ambient facts it does not itself carry: `overlap_ms` and
    /// `clamped_to_ms` (both knowable before classification: the raw
    /// overlap and the bullet-level boundary that triggered the pass) and
    /// `words_clamped` (knowable only after an `InterleavedWords` clamp
    /// actually runs; `ClampedWordCounts::default()` for the other two
    /// variants, which never touch a word). The text is READ OFF the
    /// variant, never the other way round: nothing here decides a fact the
    /// variant does not already hold. This stays a SUMMARY (counts only,
    /// via `.dropped.len()`): the full per-word records travel in the
    /// effect and trace instead, which is where a reviewer actually needs
    /// them, not duplicated into a decision-reason string.
    fn reason(
        &self,
        overlap_ms: u64,
        clamped_to_ms: u64,
        words_clamped: &ClampedWordCounts,
    ) -> String {
        match self {
            Self::CoverageOnly { hull_end_ms } => {
                let clamped_to = hull_end_ms.unwrap_or(clamped_to_ms);
                format!(
                    "end_truncated_by={overlap_ms}ms clamped_to={clamped_to}                      cuts_word_timing=false resolution=coverage_only                      cause=adjacent_utterance_overlap"
                )
            }
            Self::BoundaryFromWords {
                prev_hull_end_ms,
                next_hull_start_ms,
            } => format!(
                "end_truncated_by={overlap_ms}ms clamped_to={prev_hull_end_ms}                  cuts_word_timing=false resolution=boundary_from_words                  next_hull_start={next_hull_start_ms} cause=adjacent_utterance_overlap"
            ),
            Self::InterleavedWords => format!(
                "end_truncated_by={overlap_ms}ms clamped_to={clamped_to_ms}                  cuts_word_timing=true resolution=interleaved_words                  words_trimmed={} words_dropped={} cause=adjacent_utterance_overlap",
                words_clamped.trimmed(),
                words_clamped.dropped.len()
            ),
        }
    }
}

/// Enforce E362: strip timing from utterances whose start time is before
/// the previous utterance's start time (non-monotonic ordering).
///
/// Also truncates end-time overlaps: when utterance N's end exceeds
/// utterance N+1's start, N's end is clamped to N+1's start. The overlap can
/// originate in UTR, segmentation, prior transcript boundaries, or fresh FA;
/// the enforcement layer does not pretend to know which. A clamp that cuts a
/// retained word interval requests review, while a container-only trim does
/// not.
///
/// Special case: if the clamped end would equal or precede the utterance's
/// own start (possible when two adjacent utterances share the same start_ms
/// from overlapping UTR token ranges), the bullet is stripped entirely rather
///
/// `#[must_use]` because these records must reach durable evidence rather than
/// be produced and discarded. This is the one the attribute genuinely catches:
/// the incremental path called it BARE, for its side effect on the file, and
/// silently dropped every repair record. That form no longer compiles.
///
/// `#[cfg(test)]`: derive-only, with no production caller. Production reaches
/// monotonicity only through `FaApplied::then_finalize`, which requires the
/// kept-bullet proof.
#[cfg(test)]
pub fn enforce_monotonicity(chat_file: &mut ChatFile) -> MonotonicityResult {
    enforce_monotonicity_with_policy(chat_file, EndOverlapPolicy::ClampAllAdjacent)
}

/// Enforce start ordering and an explicit adjacent end-overlap policy, with
/// no bullet kept. `#[cfg(test)]` for the same reason as
/// [`enforce_monotonicity`].
#[cfg(test)]
pub fn enforce_monotonicity_with_policy(
    chat_file: &mut ChatFile,
    end_overlap_policy: EndOverlapPolicy,
) -> MonotonicityResult {
    enforce_monotonicity_under(chat_file, end_overlap_policy, ImposedBullets::deriving())
        .expect("the default policy refers to no given bullet")
}

/// Enforce monotonicity after the given bullets were imposed.
///
/// Under `--main-bullets keep` a kept bullet is never changed or stripped
/// here, and every guard compares the live bullet with the given one (a
/// drifted kept bullet is a [`KeptBulletError`]). The kept starts are a fixed
/// skeleton in Pass 1 (see [`enforce_start_order`]); in Pass 2 a revisable
/// bullet next to a kept one is resolved by moving only the revisable side
/// (see [`resolve_end_overlap_pair`]). Only a conflict between two kept
/// bullets is left in place, recorded as
/// [`batchalign_transform::decisions::MonotonicityStrategy::KeptBulletLeftUnresolved`]
/// with no [`MonotonicityEffect`], because nothing changed.
pub(super) fn enforce_monotonicity_under(
    chat_file: &mut ChatFile,
    end_overlap_policy: EndOverlapPolicy,
    imposed: ImposedBullets<'_>,
) -> Result<MonotonicityResult, KeptBulletError> {
    let mut decisions = Vec::new();
    let mut effects = Vec::new();
    let mut kept = KeptOverlapLedger::new(imposed);

    // Pass 1: strip utterances with non-monotonic start times.
    enforce_start_order(chat_file, imposed, &mut decisions, &mut effects)?;

    // Pass 2a: resolve every SAME-SPEAKER end overlap by walking each
    // speaker's OWN stream of bulleted utterances in file order, regardless
    // of file adjacency (2026-09-01 review, item 15). E704 (CLAN 133: a
    // speaker may not overlap themself) is defined on the speaker's OWN
    // sequence, not on physically adjacent lines: an intervening
    // other-speaker utterance (ordinary A-B-A dialogue) must not hide a
    // same-speaker overlap from resolution. Runs UNCONDITIONALLY, under
    // EITHER `EndOverlapPolicy`: the policy governs CROSS-speaker overlap
    // only (Pass 2b, below); same-speaker overlap is always resolved.
    resolve_same_speaker_stream_overlaps(chat_file, &mut decisions, &mut effects, &mut kept)?;

    // Pass 2b: additionally resolve FILE-ADJACENT pairs, honoring
    // `end_overlap_policy` for cross-speaker adjacency. `timed` is captured
    // AFTER Pass 2a so it reflects any utterance Pass 2a stripped outright
    // (a stripped utterance carries no bullet and must not enter this pass
    // at all) or moved.
    //
    // ORDER AND WHY (2026-09-01 review, item 15): Pass 2a always runs
    // first. Pass 2b runs second and, under `PreserveCrossSpeaker`, revisits
    // every pair Pass 2a already resolved that happens to ALSO be file
    // adjacent (`should_clamp` returns `true` for a same-speaker pair
    // regardless of policy) plus every genuinely cross-speaker pair (which
    // it skips under `PreserveCrossSpeaker`, resolves under
    // `ClampAllAdjacent`).
    //
    // PROOF Pass 2b cannot undo Pass 2a's resolution: every `EndOverlapResolution`
    // arm only SHRINKS the pair it resolves -- `CoverageOnly` and
    // `BoundaryFromWords` pull the earlier utterance's end DOWN toward its
    // own measured hull and/or push the later utterance's start UP toward
    // its own hull; `InterleavedWords` pulls the earlier utterance's end
    // DOWN to the later utterance's (unmoved) start. A pair Pass 2a already
    // resolved therefore satisfies `prev_bullet.end_ms <= next_start`
    // (`resolve_end_overlap_pair`'s own entry guard) by construction, so
    // Pass 2b's re-visit of that SAME pair finds no overlap and is a no-op:
    // it neither re-clamps nor reverts anything Pass 2a did. This is an
    // idempotence argument, not a policy exclusion: Pass 2b is not taught to
    // skip same-speaker pairs, because it does not need to be.
    //
    // `timed` is IDENTITY ONLY (line index, utterance ordinal, speaker), not
    // a snapshot of any timing value. A resolution below (`BoundaryFromWords`)
    // can move an utterance's bullet START, so a captured `start_ms` read
    // once here would go stale the moment that happens; every timing value
    // used below is read LIVE from `chat_file` at the point it is used,
    // which is also what lets `classify_end_overlap` see the effect of an
    // earlier pair's mutation on a later pair.
    let mut timed: Vec<(usize, UtteranceIdx, String)> = Vec::new();
    let mut utterance_ordinal = 0;
    for (line_idx, line) in chat_file.lines.iter().enumerate() {
        let Line::Utterance(utterance) = line else {
            continue;
        };
        let utterance_idx = UtteranceIdx::new(utterance_ordinal);
        utterance_ordinal += 1;
        if utterance.main.content.bullet.is_some() {
            timed.push((
                line_idx,
                utterance_idx,
                utterance.main.speaker.as_str().to_string(),
            ));
        }
    }

    for window_idx in 0..timed.len().saturating_sub(1) {
        let (prev_idx, prev_utterance_idx, prev_speaker) = &timed[window_idx];
        let (next_idx, next_utterance_idx, next_speaker) = &timed[window_idx + 1];

        if !end_overlap_policy.should_clamp(prev_speaker, next_speaker) {
            continue;
        }

        resolve_end_overlap_pair(
            chat_file,
            OverlapPairIdentity {
                prev_line_idx: *prev_idx,
                prev_utterance_idx: *prev_utterance_idx,
                next_line_idx: *next_idx,
                next_utterance_idx: *next_utterance_idx,
            },
            next_speaker,
            &mut decisions,
            &mut effects,
            &mut kept,
        )?;
    }

    Ok(MonotonicityResult {
        records: decisions,
        effects,
    })
}

/// One bulleted utterance as Pass 1 sees it.
struct StartOrderEntry {
    line_idx: usize,
    utterance_idx: UtteranceIdx,
    speaker: String,
    start_ms: u64,
    end_ms: u64,
    mutability: BulletMutability,
}

/// Pass 1: file-order start monotonicity.
///
/// Kept starts are a FIXED skeleton. A revisable bullet must start at or
/// after the greatest start accepted before it and at or before the next kept
/// start after it; one that falls outside is the one stripped, recorded as a
/// start regression (below the preceding start) or as having yielded to the
/// next kept start (above it). A kept start that regresses below an earlier
/// kept start cannot be fixed without changing a kept bullet, so it is left
/// and recorded as unresolved; it does not lower the running start, so what
/// follows is judged as though it had been stripped. With no kept bullet this
/// is exactly the established rule: strip every start below the greatest
/// preceding accepted start.
fn enforce_start_order(
    chat_file: &mut ChatFile,
    imposed: ImposedBullets<'_>,
    decisions: &mut Vec<batchalign_transform::decisions::DecisionRecord>,
    effects: &mut Vec<MonotonicityEffect>,
) -> Result<(), KeptBulletError> {
    use batchalign_transform::decisions::MonotonicityStrategy;

    let mut entries = Vec::new();
    for (line_idx, utterance_idx, utterance) in super::utterances_indexed(chat_file) {
        let Some(bullet) = utterance.main.content.bullet.as_ref() else {
            continue;
        };
        entries.push(StartOrderEntry {
            line_idx,
            utterance_idx,
            speaker: utterance.main.speaker.as_str().to_string(),
            start_ms: bullet.timing.start_ms,
            end_ms: bullet.timing.end_ms,
            mutability: imposed.mutability(utterance_idx, bullet)?,
        });
    }

    // The nearest kept entry AFTER each position. Kept bullets are never
    // stripped here, so the snapshot stays true for the whole pass.
    let mut next_kept: Vec<Option<usize>> = vec![None; entries.len()];
    let mut following_kept = None;
    for position in (0..entries.len()).rev() {
        next_kept[position] = following_kept;
        match entries[position].mutability {
            BulletMutability::ReadOnly(_) => following_kept = Some(position),
            BulletMutability::Revisable => {}
        }
    }

    let mut last_accepted: Option<usize> = None;
    let mut last_kept: Option<usize> = None;
    let mut strip = Vec::new();
    for (position, entry) in entries.iter().enumerate() {
        let below = last_accepted
            .map(|p| &entries[p])
            .filter(|previous| entry.start_ms < previous.start_ms);
        match entry.mutability {
            BulletMutability::ReadOnly(_) => {
                match last_kept
                    .map(|p| &entries[p])
                    .filter(|previous| entry.start_ms < previous.start_ms)
                {
                    Some(previous) => decisions.push(start_order_record(
                        entry,
                        MonotonicityStrategy::KeptBulletLeftUnresolved,
                        format!(
                            "non_monotonic kept_main_bullet start_ms={} previous_start_ms={} \
                             previous_line_idx={} previous_speaker={}",
                            entry.start_ms, previous.start_ms, previous.line_idx, previous.speaker
                        ),
                    )),
                    None => {
                        last_kept = Some(position);
                        last_accepted = Some(position);
                    }
                }
            }
            BulletMutability::Revisable => {
                let above = next_kept[position]
                    .map(|p| &entries[p])
                    .filter(|kept| entry.start_ms > kept.start_ms);
                match (below, above) {
                    (Some(previous), _) => {
                        decisions.push(start_order_record(
                            entry,
                            MonotonicityStrategy::TimingStripped,
                            format!(
                                "non_monotonic start_ms={} previous_start_ms={} \
                                 previous_line_idx={} previous_speaker={}",
                                entry.start_ms,
                                previous.start_ms,
                                previous.line_idx,
                                previous.speaker
                            ),
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
                        });
                        strip.push(entry.line_idx);
                    }
                    (None, Some(kept)) => {
                        decisions.push(start_order_record(
                            entry,
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
                            },
                        });
                        strip.push(entry.line_idx);
                    }
                    (None, None) => last_accepted = Some(position),
                }
            }
        }
    }

    for line_idx in strip {
        if let Line::Utterance(utterance) = &mut chat_file.lines.as_mut_slice()[line_idx] {
            strip_utterance_timing(utterance);
        }
    }
    Ok(())
}

/// The one record shape Pass 1 writes: every start-order decision needs
/// review, since each either removed timing or left a conflict in place.
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

/// Pass 2a's own sweep (2026-09-01 review, item 15): every speaker's
/// bulleted utterances, in file order, paired and resolved consecutively
/// WITHIN that speaker's own stream -- skipping over any intervening
/// other-speaker utterance, which the file-adjacent Pass 2b cannot do.
///
/// Deterministic decision order: speakers are visited in FIRST-APPEARANCE
/// order, and each speaker's own pairs are resolved in file order. Two
/// different speakers' pairs never share an utterance on both sides (a pair
/// is always same-speaker by construction here), so the relative order
/// across speakers cannot change what either speaker's resolution computes,
/// only the order their decisions/effects are recorded in.
fn resolve_same_speaker_stream_overlaps(
    chat_file: &mut ChatFile,
    decisions: &mut Vec<batchalign_transform::decisions::DecisionRecord>,
    effects: &mut Vec<MonotonicityEffect>,
    kept: &mut KeptOverlapLedger<'_>,
) -> Result<(), KeptBulletError> {
    let mut speaker_order: Vec<String> = Vec::new();
    let mut streams: HashMap<String, Vec<(usize, UtteranceIdx)>> = HashMap::new();
    for (line_idx, utterance_idx, utterance) in super::utterances_indexed(chat_file) {
        if utterance.main.content.bullet.is_none() {
            continue;
        }
        let speaker = utterance.main.speaker.as_str().to_string();
        if !streams.contains_key(&speaker) {
            speaker_order.push(speaker.clone());
        }
        streams
            .entry(speaker)
            .or_default()
            .push((line_idx, utterance_idx));
    }

    for speaker in &speaker_order {
        // SAFETY: `speaker` came from `speaker_order`, which is built from
        // the same insertion the `streams` entry itself was.
        #[allow(clippy::unwrap_used)]
        let stream = streams.get(speaker).unwrap();
        for window_idx in 0..stream.len().saturating_sub(1) {
            let (prev_idx, prev_utterance_idx) = stream[window_idx];
            let (next_idx, next_utterance_idx) = stream[window_idx + 1];
            resolve_end_overlap_pair(
                chat_file,
                OverlapPairIdentity {
                    prev_line_idx: prev_idx,
                    prev_utterance_idx,
                    next_line_idx: next_idx,
                    next_utterance_idx,
                },
                speaker,
                decisions,
                effects,
                kept,
            )?;
        }
    }
    Ok(())
}

/// What Pass 2 needs about kept bullets: the imposition proof, and which
/// pairs it has already reported as left unresolved.
///
/// The ledger exists because the two sweeps legitimately visit the same pair.
/// For a pair Pass 2 can resolve that is harmless (the first visit resolves
/// it and the second finds no overlap), but a pair of two kept bullets is
/// never resolved, so without this the file-adjacent sweep would report a
/// same-speaker pair the per-speaker sweep already reported.
struct KeptOverlapLedger<'a> {
    imposed: ImposedBullets<'a>,
    reported: std::collections::BTreeSet<(UtteranceIdx, UtteranceIdx)>,
}

impl<'a> KeptOverlapLedger<'a> {
    fn new(imposed: ImposedBullets<'a>) -> Self {
        Self {
            imposed,
            reported: std::collections::BTreeSet::new(),
        }
    }

    /// Whether this pair is being reported for the first time.
    fn first_report(&mut self, prev: UtteranceIdx, next: UtteranceIdx) -> bool {
        self.reported.insert((prev, next))
    }
}

/// Which two utterances one call to [`resolve_end_overlap_pair`] resolves.
/// Grouped into one type (2026-09-01 review, item 15) rather than four loose
/// parameters, both because the four values are one cohesive fact (a pair,
/// not four independent ones) and because a bare four-`usize`-ish argument
/// list is exactly the primitive-obsession shape this workspace's own
/// coding standards ban at a boundary like this one.
#[derive(Debug, Clone, Copy)]
struct OverlapPairIdentity {
    prev_line_idx: usize,
    prev_utterance_idx: UtteranceIdx,
    next_line_idx: usize,
    next_utterance_idx: UtteranceIdx,
}

/// The FILE-ORDER successor's start, live (2026-09-01 review, item 16): the
/// next utterance, ANY speaker, after `after_line_idx` that CURRENTLY has a
/// bullet. Walking forward live (never from a cached position) is what lets
/// this see a strip an earlier pair in the SAME sweep already performed.
///
/// This is the ONLY correct guard input for `BoundaryFromWords` moving a
/// start forward, and it replaced two WRONG per-caller substitutes: the
/// per-speaker-stream sweep used to read that speaker's own next-next
/// utterance (missing an intervening different-speaker line entirely), and
/// the file-adjacent sweep read the next entry in its own already-file-order
/// snapshot (correct by construction, but duplicated the same idea instead
/// of sharing this one). Pass 1 enforces file-order start monotonicity
/// across every speaker BEFORE Pass 2 runs; this is that SAME rule, read
/// fresh, so Pass 2 cannot undo what Pass 1 already established. Real-data
/// regression: chatter's E362 fired 1,353 times across 178 files when a
/// same-speaker `BoundaryFromWords` moved a start past an intervening
/// different-speaker line's start (2026-09-01 review, item 16).
pub(super) fn file_order_successor_start_ms(
    chat_file: &ChatFile,
    after_line_idx: usize,
) -> Option<u64> {
    chat_file
        .lines
        .iter()
        .skip(after_line_idx + 1)
        .find_map(|line| match line {
            Line::Utterance(utt) => utt.main.content.bullet.as_ref().map(|b| b.timing.start_ms),
            _ => None,
        })
}

/// Resolve one end-overlap PAIR, live, against `chat_file`. Shared
/// (2026-09-01 review, item 15) by both Pass 2 sweeps: the per-speaker
/// stream sweep, which pairs a speaker's own consecutive utterances
/// regardless of file adjacency, and the file-adjacent sweep, which
/// additionally (under `ClampAllAdjacent`) pairs any two physically
/// adjacent utterances. Both need the EXACT same per-pair resolution
/// (classification, application, decision, effect) AND the exact same
/// successor guard input, [`file_order_successor_start_ms`] (2026-09-01
/// review, item 16) -- computed HERE, not passed in, so neither caller can
/// substitute a weaker one.
fn resolve_end_overlap_pair(
    chat_file: &mut ChatFile,
    pair: OverlapPairIdentity,
    next_speaker: &str,
    decisions: &mut Vec<batchalign_transform::decisions::DecisionRecord>,
    effects: &mut Vec<MonotonicityEffect>,
    kept: &mut KeptOverlapLedger<'_>,
) -> Result<(), KeptBulletError> {
    use batchalign_transform::decisions::DecisionRecord;
    let OverlapPairIdentity {
        prev_line_idx: prev_idx,
        prev_utterance_idx,
        next_line_idx: next_idx,
        next_utterance_idx,
    } = pair;

    // `next`'s CURRENT bullet (its start may have moved via an earlier
    // pair's `BoundaryFromWords` in the SAME sweep), read live.
    let Line::Utterance(next_utt_ref) = &chat_file.lines.as_slice()[next_idx] else {
        return Ok(());
    };
    let Some(next_bullet) = next_utt_ref.main.content.bullet.as_ref() else {
        return Ok(());
    };
    let next_start = next_bullet.timing.start_ms;
    let next_mutability = kept.imposed.mutability(next_utterance_idx, next_bullet)?;

    // Read what is true of the previous utterance's bullet before touching
    // anything: whether an overlap exists at all is decided read-only, and
    // prev/next share one Vec so only one of them can be borrowed mutably
    // at a time.
    let Line::Utterance(prev_utt_ref) = &chat_file.lines.as_slice()[prev_idx] else {
        return Ok(());
    };
    let Some(prev_bullet) = prev_utt_ref.main.content.bullet.as_ref() else {
        return Ok(());
    };
    if prev_bullet.timing.end_ms <= next_start {
        return Ok(());
    }
    let prev_mutability = kept.imposed.mutability(prev_utterance_idx, prev_bullet)?;
    let facts = OverlapFacts {
        edge: OverlapEdge {
            line_idx: prev_idx,
            utterance_idx: prev_utterance_idx,
            speaker: prev_utt_ref.main.speaker.as_str().to_string(),
            original_end_ms: prev_bullet.timing.end_ms,
            next_line_idx: next_idx,
            next_utterance_idx,
            next_speaker: next_speaker.to_string(),
        },
        prev_start_ms: prev_bullet.timing.start_ms,
        next_start_ms: next_start,
    };

    // Every resolution of a revisable `prev` moves or strips `prev`'s
    // bullet, and `BoundaryFromWords` also moves `next`'s start; a kept
    // `prev` is resolved, if at all, by moving `next`. Written as the full
    // 2x2 so a new mutability cannot fall into someone else's arm.
    match (prev_mutability, next_mutability) {
        (BulletMutability::Revisable, BulletMutability::Revisable) => {
            resolve_revisable_prev(chat_file, facts, NextStart::Movable, decisions, effects);
        }
        (BulletMutability::Revisable, BulletMutability::ReadOnly(_)) => {
            resolve_revisable_prev(chat_file, facts, NextStart::Kept, decisions, effects);
        }
        (BulletMutability::ReadOnly(kept_prev), BulletMutability::Revisable) => {
            yield_next_to_kept_end(chat_file, facts, kept_prev, decisions, effects);
        }
        (BulletMutability::ReadOnly(_), BulletMutability::ReadOnly(_)) => {
            if kept.first_report(prev_utterance_idx, next_utterance_idx) {
                let edge = &facts.edge;
                decisions.push(DecisionRecord::new_and_trace(
                    edge.line_idx,
                    edge.speaker.clone(),
                    batchalign_transform::decisions::DecisionStrategy::Monotonicity(
                        batchalign_transform::decisions::MonotonicityStrategy::KeptBulletLeftUnresolved,
                    ),
                    format!(
                        "end_overlap kept_main_bullet start_ms={} original_end={} \
                         next_start={} overlap={}ms next_line_idx={} next_speaker={} \
                         cause=adjacent_kept_bullets_overlap",
                        facts.prev_start_ms,
                        edge.original_end_ms,
                        facts.next_start_ms,
                        facts.overlap_ms(),
                        edge.next_line_idx,
                        edge.next_speaker
                    ),
                    // Cross-speaker overlap is ordinary conversation and valid
                    // CHAT; a speaker overlapping themself is what E704
                    // rejects, so only that is left for a person.
                    edge.speaker == edge.next_speaker,
                ));
            }
        }
    }
    Ok(())
}

/// The facts one overlapping pair was resolved from, read before any change.
struct OverlapFacts {
    /// The pair's identities and the previous utterance's original end.
    edge: OverlapEdge,
    /// The previous utterance's bullet start.
    prev_start_ms: u64,
    /// The next utterance's bullet start.
    next_start_ms: u64,
}

impl OverlapFacts {
    fn overlap_ms(&self) -> u64 {
        self.edge.original_end_ms - self.next_start_ms
    }
}

/// Whether the next utterance's start may move when resolving a pair whose
/// previous bullet is revisable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NextStart {
    /// The next bullet is revisable: every resolution is available.
    Movable,
    /// The next bullet is kept: nothing may move its start.
    Kept,
}

/// Resolve an overlap whose PREVIOUS bullet is revisable, cutting or
/// stripping only that side (and, when `next` is movable, possibly moving
/// `next`'s start to its word hull).
fn resolve_revisable_prev(
    chat_file: &mut ChatFile,
    facts: OverlapFacts,
    next: NextStart,
    decisions: &mut Vec<batchalign_transform::decisions::DecisionRecord>,
    effects: &mut Vec<MonotonicityEffect>,
) {
    use batchalign_transform::decisions::DecisionRecord;
    let OverlapFacts {
        edge,
        prev_start_ms: start_ms,
        next_start_ms: next_start,
    } = facts;
    let (prev_idx, next_idx) = (edge.line_idx, edge.next_line_idx);
    let original_end = edge.original_end_ms;
    let overlap_ms = original_end - next_start;
    let speaker = edge.speaker.clone();

    if next_start <= start_ms {
        // Clamping would produce a zero-or-negative-duration bullet
        // (next_start <= prev.start), which fails E362. Strip the
        // bullet entirely; untimed is safer than invalid. The cause says
        // why the next start could not give way: identical starts from
        // overlapping UTR token ranges, or a kept next bullet.
        let cause = match next {
            NextStart::Movable => "utr_identical_start_times",
            NextStart::Kept => "yielded_to_kept_start",
        };
        decisions.push(DecisionRecord::new_and_trace(
            prev_idx,
            speaker.clone(),
            batchalign_transform::decisions::DecisionStrategy::Monotonicity(
                batchalign_transform::decisions::MonotonicityStrategy::TimingStripped,
            ),
            format!(
                "zero_duration_clamp original_end={original_end} \
                 next_start={next_start} start_ms={start_ms} \
                 cause={cause}"
            ),
            true,
        ));
        effects.push(MonotonicityEffect::ZeroDurationClampStripped {
            line_idx: prev_idx,
            utterance_idx: edge.utterance_idx,
            speaker,
            start_ms,
            original_end_ms: original_end,
            next_start_ms: next_start,
            next_line_idx: next_idx,
            next_utterance_idx: edge.next_utterance_idx,
            next_speaker: edge.next_speaker,
        });
        if let Line::Utterance(prev_utt) = &mut chat_file.lines.as_mut_slice()[prev_idx] {
            strip_utterance_timing(prev_utt);
        }
        return;
    }

    // Classify from what is MEASURED (word timings), never from the
    // coverage-extended bullet alone: see `EndOverlapResolution`.
    let prev_hull_end_ms = match &chat_file.lines.as_slice()[prev_idx] {
        Line::Utterance(prev_utt_ref) => furthest_word_timing_end(prev_utt_ref),
        _ => None,
    };
    let next_hull_start_ms = match &chat_file.lines.as_slice()[next_idx] {
        Line::Utterance(next_utt_ref) => earliest_word_timing_start(next_utt_ref),
        _ => None,
    };
    // The FILE-ORDER successor, live, any speaker (2026-09-01 review, item
    // 16): the only correct guard input, regardless of which sweep formed
    // this pair. See `file_order_successor_start_ms`'s own doc.
    let next_successor_start_ms = file_order_successor_start_ms(chat_file, next_idx);
    let classified = classify_end_overlap(
        prev_hull_end_ms,
        next_start,
        next_hull_start_ms,
        next_successor_start_ms,
    );
    let resolution = match next {
        NextStart::Movable => classified,
        NextStart::Kept => classified.keeping_next_start(),
    };

    match resolution {
        EndOverlapResolution::CoverageOnly { hull_end_ms } => {
            let clamped_to_ms = hull_end_ms.unwrap_or(next_start);
            decisions.push(DecisionRecord::new_and_trace(
                prev_idx,
                speaker,
                batchalign_transform::decisions::DecisionStrategy::Monotonicity(
                    resolution.strategy(),
                ),
                resolution.reason(overlap_ms, next_start, &ClampedWordCounts::default()),
                resolution.needs_review(),
            ));
            effects.push(MonotonicityEffect::EndClampedCoverageOnly {
                edge,
                clamped_to_ms,
            });
            if let Line::Utterance(prev_utt) = &mut chat_file.lines.as_mut_slice()[prev_idx]
                && let Some(bullet) = prev_utt.main.content.bullet.as_mut()
            {
                bullet.timing.end_ms = clamped_to_ms;
            }
        }
        EndOverlapResolution::BoundaryFromWords {
            prev_hull_end_ms,
            next_hull_start_ms,
        } => {
            decisions.push(DecisionRecord::new_and_trace(
                prev_idx,
                speaker,
                batchalign_transform::decisions::DecisionStrategy::Monotonicity(
                    resolution.strategy(),
                ),
                resolution.reason(overlap_ms, next_start, &ClampedWordCounts::default()),
                resolution.needs_review(),
            ));
            effects.push(MonotonicityEffect::EndClampedBoundaryFromWords {
                edge,
                prev_hull_end_ms,
                next_hull_start_ms,
            });
            if let Line::Utterance(prev_utt) = &mut chat_file.lines.as_mut_slice()[prev_idx]
                && let Some(bullet) = prev_utt.main.content.bullet.as_mut()
            {
                bullet.timing.end_ms = prev_hull_end_ms;
            }
            if let Line::Utterance(next_utt) = &mut chat_file.lines.as_mut_slice()[next_idx]
                && let Some(bullet) = next_utt.main.content.bullet.as_mut()
            {
                bullet.timing.start_ms = next_hull_start_ms;
            }
        }
        EndOverlapResolution::InterleavedWords => {
            let mut words_clamped = ClampedWordCounts::default();
            if let Line::Utterance(prev_utt) = &mut chat_file.lines.as_mut_slice()[prev_idx] {
                if let Some(bullet) = prev_utt.main.content.bullet.as_mut() {
                    bullet.timing.end_ms = next_start;
                }
                words_clamped = clamp_words_past_bound(prev_utt, next_start);
            }
            decisions.push(DecisionRecord::new_and_trace(
                prev_idx,
                speaker,
                batchalign_transform::decisions::DecisionStrategy::Monotonicity(
                    resolution.strategy(),
                ),
                resolution.reason(overlap_ms, next_start, &words_clamped),
                resolution.needs_review(),
            ));
            effects.push(MonotonicityEffect::EndClampedInterleavedWords {
                edge,
                clamped_to_ms: next_start,
                words_trimmed: words_clamped.trimmed(),
                words_dropped: words_clamped.dropped,
            });
        }
    }
}

/// How a revisable NEXT bullet gives way to a kept previous bullet's end.
///
/// The mirror of [`EndOverlapResolution::keeping_next_start`]: there the
/// kept side is `next` and only `prev` may change; here the kept side is
/// `prev`, so only `next`'s start may move, and only forward.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KeptEndYield {
    /// `next`'s words already start at or after the kept end: move only its
    /// bullet start there.
    StartToKeptEnd,
    /// `next`'s leading words reach back before the kept end: move its start
    /// and cut those words to the kept end.
    LeadingWordsClamped,
    /// `next` ends at or before the kept end, or moving its start would pass
    /// its own file-order successor's start: no valid bullet remains, strip.
    Strip,
}

impl KeptEndYield {
    /// Classify from the kept end, `next`'s end, `next`'s measured first word
    /// start, and `next`'s file-order successor's live start.
    fn keeping_prev_end(
        kept_end_ms: u64,
        next_end_ms: u64,
        next_hull_start_ms: Option<u64>,
        next_successor_start_ms: Option<u64>,
    ) -> Self {
        let passes_successor = match next_successor_start_ms {
            Some(successor_start_ms) => successor_start_ms < kept_end_ms,
            None => false,
        };
        match (next_end_ms > kept_end_ms, passes_successor) {
            (false, _) | (true, true) => Self::Strip,
            (true, false) => match next_hull_start_ms {
                Some(hull_start_ms) if hull_start_ms < kept_end_ms => Self::LeadingWordsClamped,
                Some(_) | None => Self::StartToKeptEnd,
            },
        }
    }
}

/// Resolve an overlap whose PREVIOUS bullet is kept and whose next is
/// revisable, by moving only the next utterance (see [`KeptEndYield`]).
fn yield_next_to_kept_end(
    chat_file: &mut ChatFile,
    facts: OverlapFacts,
    kept_prev: KeptBullet,
    decisions: &mut Vec<batchalign_transform::decisions::DecisionRecord>,
    effects: &mut Vec<MonotonicityEffect>,
) {
    use batchalign_transform::decisions::{DecisionRecord, DecisionStrategy, MonotonicityStrategy};
    let OverlapFacts {
        edge,
        next_start_ms: next_start,
        ..
    } = facts;
    let next_idx = edge.next_line_idx;
    let kept_end = kept_prev.end_ms();
    let Line::Utterance(next_utt_ref) = &chat_file.lines.as_slice()[next_idx] else {
        return;
    };
    let Some(next_end) = next_utt_ref
        .main
        .content
        .bullet
        .as_ref()
        .map(|bullet| bullet.timing.end_ms)
    else {
        return;
    };
    let yielded = KeptEndYield::keeping_prev_end(
        kept_end,
        next_end,
        earliest_word_timing_start(next_utt_ref),
        file_order_successor_start_ms(chat_file, next_idx),
    );
    let Line::Utterance(next_utt) = &mut chat_file.lines.as_mut_slice()[next_idx] else {
        return;
    };
    let (strategy, outcome, needs_review, cause) = match yielded {
        KeptEndYield::Strip => {
            strip_utterance_timing(next_utt);
            (
                MonotonicityStrategy::TimingStripped,
                KeptBulletYield::Stripped {
                    start_ms: next_start,
                    end_ms: next_end,
                },
                true,
                "no_room_after_kept_end",
            )
        }
        KeptEndYield::StartToKeptEnd | KeptEndYield::LeadingWordsClamped => {
            if let Some(bullet) = next_utt.main.content.bullet.as_mut() {
                bullet.timing.start_ms = kept_end;
            }
            let words = match yielded {
                KeptEndYield::LeadingWordsClamped => {
                    clamp_words_within(next_utt, TimeSpan::new(kept_end, next_end))
                }
                KeptEndYield::StartToKeptEnd | KeptEndYield::Strip => ClampedWordCounts::default(),
            };
            let cut_any = words.trimmed() > 0 || !words.dropped.is_empty();
            (
                MonotonicityStrategy::YieldedToKeptBullet,
                KeptBulletYield::StartMoved {
                    from_ms: next_start,
                    to_ms: kept_end,
                    words_trimmed: words.trimmed(),
                    words_dropped: words.dropped,
                },
                cut_any,
                "start_moved_to_kept_end",
            )
        }
    };
    decisions.push(DecisionRecord::new_and_trace(
        next_idx,
        edge.next_speaker.clone(),
        DecisionStrategy::Monotonicity(strategy),
        format!(
            "yielded_to_kept_end start_ms={next_start} end_ms={next_end} kept_end={kept_end} \
             kept_line_idx={} kept_speaker={} cause={cause}",
            edge.line_idx, edge.speaker
        ),
        needs_review,
    ));
    effects.push(MonotonicityEffect::YieldedToKeptBullet {
        line_idx: next_idx,
        utterance_idx: edge.next_utterance_idx,
        speaker: edge.next_speaker,
        kept_line_idx: edge.line_idx,
        kept_utterance_idx: edge.utterance_idx,
        kept_speaker: edge.speaker,
        kept_boundary_ms: kept_end,
        outcome,
    });
}

/// Strip `%wor` tiers from utterances whose bullets were removed by
/// `enforce_monotonicity`.
///
/// ## Why this is necessary
///
/// `enforce_monotonicity` strips a main-tier bullet when it is non-monotonic
/// (backward relative to the previous utterance).  However, if the utterance
/// also carries a `%wor` tier with the same backward timestamps, the NEXT
/// re-run will enter the `has_reusable_wor_timing` fast path, call
/// `refresh_existing_alignment`, and reconstruct the backward bullet from
/// the stale `%wor` data, reintroducing the E362 violation.
///
/// This function removes the `%wor` tier from every utterance that
/// `enforce_monotonicity` stripped, breaking the re-run cycle:
///
/// ```text
/// bad %wor → fast path → backward bullet → enforce strips bullet
///          → strip_wor removes %wor → next run skips fast path
///          → full FA re-aligns → correct timing
/// ```
///
/// Call this immediately after `enforce_monotonicity` wherever the
/// `has_reusable_wor_timing` fast path is used:
/// - `run_fa_from_ast` fast-path return in `batchalign/src/fa/mod.rs`
///
/// `decisions` must be the slice returned by `enforce_monotonicity`.
/// Only decisions with `module == DecisionModule::Monotonicity` and
/// `strategy == "timing_stripped"` are processed; end-clamped utterances
/// keep their `%wor` because their timing is still monotonic.
pub fn strip_wor_from_monotonicity_stripped_utterances(
    chat_file: &mut ChatFile,
    decisions: &MonotonicityResult,
) {
    use batchalign_transform::decisions::{DecisionStrategy, MonotonicityStrategy};

    // Collect the line indices of utterances that had their bullets stripped
    // (as opposed to end-clamped, which still have valid timing).
    // Typically 0-2 utterances, so a small Vec is cheaper than a HashSet.
    let stripped: Vec<usize> = decisions
        .records()
        .iter()
        .filter(|d| {
            matches!(
                d.strategy,
                DecisionStrategy::Monotonicity(MonotonicityStrategy::TimingStripped)
            )
        })
        .map(|d| d.line_idx.raw())
        .collect();

    if stripped.is_empty() {
        return;
    }

    for (line_idx, line) in chat_file.lines.as_mut_slice().iter_mut().enumerate() {
        if !stripped.contains(&line_idx) {
            continue;
        }
        let Line::Utterance(utterance) = line else {
            continue;
        };
        // Remove the %wor tier.  The utterance will be re-aligned on the next
        // run through the full FA path rather than the fast path.
        super::remove_wor_tier(utterance);
    }
}

/// chatter's own E704 tolerance (CLAN 133: a speaker may not overlap
/// themself), as ONE typed, shared value (2026-09-01 review, item 15): a
/// same-speaker overlap this size or smaller is within chatter's own
/// leniency. Before this constant, `repair.rs`'s
/// `BOUNDARY_AVERAGING_THRESHOLD_MS` independently hand-typed the SAME
/// number with no shared owner (shape D: knowledge duplicated with no
/// owner); it now reads this constant instead of restating it.
pub(super) const E704_TOLERANCE_MS: u64 = 500;

// ---------------------------------------------------------------------------
// Timing stripping helpers
// ---------------------------------------------------------------------------

/// Strip all timing information from utterance content items.
///
/// Removes `InternalBullet` items and clears `inline_bullet` from all words.
pub fn strip_timing_from_content(items: &mut TierContentItems) {
    items.retain(|item| !matches!(item, UtteranceContent::InternalBullet(_)));

    for item in items.as_mut_slice().iter_mut() {
        match item {
            UtteranceContent::Word(w) => {
                w.inline_bullet = None;
            }
            UtteranceContent::AnnotatedWord(aw) => {
                aw.inner.inline_bullet = None;
            }
            UtteranceContent::ReplacedWord(rw) => {
                rw.word.inline_bullet = None;
            }
            UtteranceContent::Group(g) => {
                strip_timing_from_bracketed(&mut g.content.content);
            }
            UtteranceContent::AnnotatedGroup(ag) => {
                strip_timing_from_bracketed(&mut ag.inner.content.content);
            }
            _ => {}
        }
    }
}

/// Remove parsed internal bullet tokens while preserving `Word.inline_bullet`.
///
/// This is used by the cheap rerun path after `%wor` timing is copied back to
/// main-tier words. Without this cleanup the serializer would emit both the
/// old parsed bullet tokens and the refreshed word-level bullets.
fn strip_internal_bullet_tokens(items: &mut TierContentItems) {
    items.retain(|item| !matches!(item, UtteranceContent::InternalBullet(_)));

    for item in items.as_mut_slice().iter_mut() {
        match item {
            UtteranceContent::Group(group) => {
                strip_internal_bullet_tokens_bracketed(&mut group.content.content);
            }
            UtteranceContent::AnnotatedGroup(group) => {
                strip_internal_bullet_tokens_bracketed(&mut group.inner.content.content);
            }
            _ => {}
        }
    }
}

fn strip_internal_bullet_tokens_bracketed(items: &mut BracketedItems) {
    items.retain(|item| !matches!(item, BracketedItem::InternalBullet(_)));

    for item in items.as_mut_slice().iter_mut() {
        match item {
            BracketedItem::AnnotatedGroup(group) => {
                strip_internal_bullet_tokens_bracketed(&mut group.inner.content.content);
            }
            BracketedItem::PhoGroup(group) => {
                strip_internal_bullet_tokens_bracketed(&mut group.content.content);
            }
            BracketedItem::SinGroup(group) => {
                strip_internal_bullet_tokens_bracketed(&mut group.content.content);
            }
            BracketedItem::Quotation(group) => {
                strip_internal_bullet_tokens_bracketed(&mut group.content.content);
            }
            _ => {}
        }
    }
}

fn strip_timing_from_bracketed(items: &mut BracketedItems) {
    items.retain(|item| !matches!(item, BracketedItem::InternalBullet(_)));

    for item in items.as_mut_slice().iter_mut() {
        match item {
            BracketedItem::Word(w) => {
                w.inline_bullet = None;
            }
            BracketedItem::AnnotatedWord(aw) => {
                aw.inner.inline_bullet = None;
            }
            BracketedItem::AnnotatedGroup(ag) => {
                strip_timing_from_bracketed(&mut ag.inner.content.content);
            }
            _ => {}
        }
    }
}

/// Collect a flat timing vector for main-tier Wor-alignable words by aligning
/// the existing `%wor` tier back onto the main tier.
fn collect_wor_backed_timings(utterance: &Utterance) -> Option<Vec<Option<WordTiming>>> {
    const MAX_REUSABLE_WOR_WORD_DURATION_PROPORTION: f64 = 0.4;
    const MIN_WORDS_FOR_DOMINANCE_CHECK: usize = 3;
    const MIN_REUSABLE_WOR_WORD_DURATION_MS: u64 = 40;

    let wor = utterance.wor_tier()?;
    let count_matched = match bind_wor_timing(&utterance.main, Some(wor)) {
        WorTimingBinding::CountMatched(count_matched) => count_matched,
        WorTimingBinding::Missing(_) | WorTimingBinding::Drifted(_) => return None,
    };
    let corroborated = match corroborate_wor_timing(count_matched) {
        WorTimingCorrespondence::Corroborated(corroborated) => corroborated,
        WorTimingCorrespondence::Uncorroborated(_) => return None,
    };
    let complete = match assess_wor_timing_sequence(corroborated) {
        WorTimingSequence::Complete(complete) => complete,
        WorTimingSequence::Empty(_) | WorTimingSequence::Rejected(_) => return None,
    };
    let timings: Vec<Option<WordTiming>> = complete
        .slots()
        .iter()
        .map(|slot| Some(WordTiming::from_complete_wor_slot(slot)))
        .collect();

    if timings
        .iter()
        .flatten()
        .any(|span| span.duration_ms() < MIN_REUSABLE_WOR_WORD_DURATION_MS)
    {
        return None;
    }

    if timings.len() >= MIN_WORDS_FOR_DOMINANCE_CHECK {
        let mut first_start = None;
        let mut last_end = None;
        let mut max_duration_ms = 0u64;
        for span in timings.iter().flatten() {
            if first_start.is_none_or(|start| span.start_ms < start) {
                first_start = Some(span.start_ms);
            }
            if last_end.is_none_or(|end| span.end_ms > end) {
                last_end = Some(span.end_ms);
            }
            max_duration_ms = max_duration_ms.max(span.duration_ms());
        }
        let utterance_span_ms = last_end?.saturating_sub(first_start?);
        if utterance_span_ms > 0
            && (max_duration_ms as f64 / utterance_span_ms as f64)
                > MAX_REUSABLE_WOR_WORD_DURATION_PROPORTION
        {
            return None;
        }
    }

    Some(timings)
}

pub(super) fn collect_wor_backed_span(utterance: &Utterance) -> Option<WordTiming> {
    let timings = collect_wor_backed_timings(utterance)?;
    let mut first_start = None;
    let mut last_end = None;
    for span in timings.iter().flatten() {
        if first_start.is_none_or(|start| span.start_ms < start) {
            first_start = Some(span.start_ms);
        }
        if last_end.is_none_or(|end| span.end_ms > end) {
            last_end = Some(span.end_ms);
        }
    }
    // The extremes of a set of spans: the min and max of other numbers, so it
    // must not read as measured. `MergedFromParts` is the variant for exactly
    // this and says so in its own doc.
    let parts = timings.iter().flatten().count();
    WordTiming::new(
        first_start?,
        last_end?,
        Origin::MergedFromParts { parts },
        Origin::MergedFromParts { parts },
    )
}

/// Strip timing and %wor from a single utterance.
pub(super) fn strip_utterance_timing(utt: &mut Utterance) -> StrippedTiming {
    utt.main.content.bullet = None;
    strip_timing_from_content(&mut utt.main.content.content);
    // Remove %wor tiers.
    utt.dependent_tiers
        .retain(|t| !matches!(t.tier, DependentTier::Wor(_)));
    StrippedTiming
}

/// Every source of structured decision records one FA run can produce.
///
/// # Why a struct rather than two hand-built vectors
///
/// Both FA paths assembled this list by hand, in an order held equal by a
/// comment reading "same decisions, and in the same order, as the full path".
/// That comment was already untrue when it was written: the full path extends
/// five sources and the incremental path four, because incremental never runs
/// narrow-bullet rescue. The difference is legitimate, which is exactly why a
/// comment is the wrong thing to hold it: nothing would have noticed when the
/// sets diverged for a BAD reason, and a sixth source added to one path only
/// would make durable evidence silently thinner on one path.
///
/// Built with a struct literal on purpose. Adding a field breaks BOTH paths at
/// compile time, and the incremental path states `rescue: Vec::new()` outright
/// rather than leaving its absence to be inferred from a list that stops early.
pub struct FaDecisions {
    /// Bullets pre-expanded before grouping, where transcribe under-budgeted.
    pub rescue: Vec<batchalign_transform::decisions::DecisionRecord>,
    /// Utterances grouping refused to place. These reach the transcript with no
    /// timing at all, so a reviewer needs them more than any adjustment record.
    pub unplaceable: Vec<batchalign_transform::decisions::DecisionRecord>,
    /// Injection, optional repair, and monotonicity in the required order.
    pub finalized: FaFinalized,
}

/// Proof that this run's decision records were written to the CHAT file.
///
/// The private field prevents callers from manufacturing the state. Consuming
/// it is the only way to move the exact written records into durable evidence,
/// so transcript review tiers and JSON evidence cannot silently diverge.
#[must_use = "written FA decisions must be retained in the evidence trace"]
pub struct WrittenFaDecisions {
    records: Vec<batchalign_transform::decisions::DecisionRecord>,
    timing_effects: Vec<MonotonicityEffect>,
}

impl WrittenFaDecisions {
    /// Consume the proof and return the records that were written.
    pub fn into_evidence(
        self,
    ) -> (
        Vec<batchalign_transform::decisions::DecisionRecord>,
        Vec<MonotonicityEffect>,
    ) {
        (self.records, self.timing_effects)
    }
}

impl FaDecisions {
    /// Assemble a run that made no fresh FA injection but still changed or
    /// refused timing before/while enforcing monotonicity.
    ///
    /// The all-`%wor` fast path and a grouping-empty path are legitimate
    /// no-injection runs. They still need the same write-before-evidence
    /// typestate as a full run; this constructor is the only way to create the
    /// otherwise-private empty [`FaFinalized`] postprocess state. Callers must
    /// obtain that state through [`finalize_without_injection`].
    pub fn without_injection(
        rescue: Vec<batchalign_transform::decisions::DecisionRecord>,
        unplaceable: Vec<batchalign_transform::decisions::DecisionRecord>,
        finalized: FaFinalized,
    ) -> Self {
        Self {
            rescue,
            unplaceable,
            finalized,
        }
    }

    /// The records and numeric timing effects in the order they must appear.
    #[must_use]
    fn into_parts(
        self,
    ) -> (
        Vec<batchalign_transform::decisions::DecisionRecord>,
        Vec<MonotonicityEffect>,
    ) {
        let Self {
            rescue,
            unplaceable,
            finalized:
                FaFinalized {
                    ordered:
                        FaOrdered {
                            postprocess,
                            monotonicity,
                        },
                    repair,
                    repair_stats: _,
                    kept_bullets: _,
                },
        } = self;
        let (monotonicity, timing_effects) = monotonicity.into_parts();
        let mut records = Vec::with_capacity(
            rescue.len()
                + unplaceable.len()
                + postprocess.len()
                + monotonicity.len()
                + repair.len(),
        );
        records.extend(rescue);
        records.extend(unplaceable);
        records.extend(postprocess);
        records.extend(repair);
        records.extend(monotonicity);
        (records, timing_effects)
    }
}

/// Finalize this run's decision provenance: strip legacy CHAT tiers and retain
/// the typed records for structured evidence.
///
/// # Why the strip is unconditional and separate
///
/// Review-level compatibility input does not reach this operation, so no
/// caller can accidentally make it authorize CHAT-tier generation. Calling
/// the policy owner unconditionally guarantees stale `%xalign` and `%xrev`
/// tiers are removed even when this run produced no decisions.
pub fn retain_decision_evidence(
    chat_file: &mut ChatFile,
    decisions: FaDecisions,
) -> WrittenFaDecisions {
    let (records, timing_effects) = decisions.into_parts();
    batchalign_transform::decisions::strip_decision_tiers(chat_file);
    WrittenFaDecisions {
        records,
        timing_effects,
    }
}
