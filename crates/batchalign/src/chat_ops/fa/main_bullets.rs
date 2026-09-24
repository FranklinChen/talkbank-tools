//! Main-tier utterance bullets the input carried, and whether a projection
//! may change them (`align --main-bullets {derive,keep}`).
//!
//! # Why the bullets are captured at the parse
//!
//! By the time FA evidence is projected, the working document's bullets are no
//! longer the input's. Several passes rewrite them first, for reasons of their
//! own: narrow-bullet rescue widens an under-budgeted bullet into the following
//! gap, edge-filler expansion widens a bullet to reach a filler, the partial
//! `%wor` refresh unions a bullet with its refreshed words, and two-pass UTR
//! writes an ordinary authoritative bullet onto an utterance the input left
//! UNBULLETED. No field on a `Bullet` tells those apart from a given one
//! (`BulletSource` cannot: the two-pass writer uses `Authoritative`). So the
//! bullets are captured by [`MainBulletAuthority::bind`] at the single parse in
//! the align dispatch, before any of those passes, and travel with the model.
//!
//! # The graph
//!
//! ```text
//! input bullet --classify--> GivenMainBullet::{Extent, Empty, Backward}
//!     Backward: bind refuses the file (BackwardGivenBullet)
//!     Extent | Empty: KeptBullet, read-only for the run
//! input without a bullet: Given::Unbulleted, read through the document's ONE
//! AbsencePolicy (a binding cannot mix the two):
//!     keep  (AbsencePolicy::Derive): derives one from its words as by default
//!     exact (AbsencePolicy::Keep):   stays without one; its words lose timing
//! MainBulletPolicy + input --MainBulletAuthority::bind--> MainBulletAuthority
//! FaProjectionPolicy + MainBulletAuthority --FaProjection::new--> FaProjection
//! MainBulletAuthority --impose(chat)--> Imposition
//!     = ImposedBullets (proof: given bullets restored, words clamped)
//!     + KeptWordCuts   (utterances whose %wor must be regenerated)
//!     + records
//! ImposedBullets --required by--> bullet repair, monotonicity
//! MainBulletAuthority --verify_held(chat)--> KeptBulletsHeld | KeptBulletError
//! ```
//!
//! Repair and monotonicity cannot run without an [`ImposedBullets`], and their
//! guards compare the LIVE bullet with the [`KeptBullet`] they carry, so a kept
//! bullet that drifted is an error at the first phase to see it rather than a
//! silent wrong answer. Those phases only ever ask about an utterance that HAS
//! a live bullet, so a kept absence is never one of their read-only nodes: an
//! absence that gained a bullet is [`KeptBulletError::GainedBullet`] instead.
//! The phases before `impose` ask [`MainBulletAuthority::given_mutability`],
//! whose [`GivenMutability::ReadOnly`] covers both, so the injection writes no
//! bullet onto a kept absence in the first place. The run ends with
//! [`MainBulletAuthority::verify_held`], whose [`KeptBulletsHeld`] proof
//! `FaFinalized` carries.

use std::fmt::Write as _;

use batchalign_transform::decisions::{DecisionRecord, DecisionStrategy, FaStrategy};
use talkbank_model::UtteranceIdx;
use talkbank_model::model::{Bullet, ChatFile};

use super::orchestrate::{ClampedWordCounts, WordTier, clamp_words_within};
use super::{
    FaProjectionPolicy, MainBulletPolicy, TimeSpan, utterances_indexed, utterances_indexed_mut,
};

/// What one input bullet is, before the run decides what to do with it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum GivenMainBullet {
    /// A positive extent: kept exactly.
    Extent { start_ms: u64, end_ms: u64 },
    /// Start equals end. Kept exactly as written, with every word untimed
    /// (nothing fits in no time), and recorded.
    Empty { at_ms: u64 },
    /// End before start. No word can be placed against it and it cannot be
    /// kept "exactly" as a valid bullet, so the file is refused at bind.
    Backward { start_ms: u64, end_ms: u64 },
}

impl GivenMainBullet {
    fn classify(bullet: &Bullet) -> Self {
        let (start_ms, end_ms) = (bullet.timing.start_ms, bullet.timing.end_ms);
        match start_ms.cmp(&end_ms) {
            std::cmp::Ordering::Less => Self::Extent { start_ms, end_ms },
            std::cmp::Ordering::Equal => Self::Empty { at_ms: start_ms },
            std::cmp::Ordering::Greater => Self::Backward { start_ms, end_ms },
        }
    }
}

/// A given bullet the run keeps: the two admissible [`GivenMainBullet`] nodes.
///
/// Built only by [`MainBulletAuthority::bind`], so a value is always a bullet
/// read off the input. `Backward` has no variant here, which is what makes a
/// kept backward bullet unrepresentable rather than merely checked for. The
/// variants live in the private [`Kept`], so the sibling phases that receive a
/// `KeptBullet` can read it but cannot build one from raw milliseconds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct KeptBullet(Kept);

/// The two shapes of a [`KeptBullet`]; private to this module.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kept {
    /// A positive extent.
    Extent { start_ms: u64, end_ms: u64 },
    /// A zero-length bullet, kept as written.
    Empty { at_ms: u64 },
}

impl KeptBullet {
    /// Start, in milliseconds.
    pub(super) fn start_ms(self) -> u64 {
        match self.0 {
            Kept::Extent { start_ms, .. } => start_ms,
            Kept::Empty { at_ms } => at_ms,
        }
    }

    /// End, in milliseconds.
    pub(super) fn end_ms(self) -> u64 {
        match self.0 {
            Kept::Extent { end_ms, .. } => end_ms,
            Kept::Empty { at_ms } => at_ms,
        }
    }

    /// The span words are clamped into. Empty for an `Empty` bullet, so every
    /// timed word falls outside it.
    fn span(self) -> TimeSpan {
        TimeSpan::new(self.start_ms(), self.end_ms())
    }

    /// The bullet as written back; `Bullet::new` serializes the same pair.
    fn to_bullet(self) -> Bullet {
        Bullet::new(self.start_ms(), self.end_ms())
    }

    /// Whether the live bullet is exactly this one.
    fn is_live(self, live: Option<&Bullet>) -> bool {
        match live {
            Some(bullet) => {
                bullet.timing.start_ms == self.start_ms() && bullet.timing.end_ms == self.end_ms()
            }
            None => false,
        }
    }
}

/// What the input carried for one utterance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Given {
    /// No bullet. What that means is the document's [`AbsencePolicy`].
    Unbulleted,
    /// This bullet, which the run keeps.
    Kept(KeptBullet),
}

/// What the run does with an utterance the input left without a bullet. One
/// value per document, held beside the slots rather than in each, so a
/// binding cannot mix the two meanings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AbsencePolicy {
    /// `keep`: the utterance derives a bullet from its words, as by default.
    Derive,
    /// `exact`: the utterance stays without a bullet and its words are untimed.
    Keep,
}

/// What the run does with one utterance's bullet: [`Given`] read through the
/// document's [`AbsencePolicy`]. Computed by [`GivenMainBullets::slot`], never
/// stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GivenSlot {
    /// No bullet was given and the run derives one.
    Unbulleted,
    /// No bullet was given and the run keeps that absence.
    KeptAbsent,
    /// The input gave this bullet and the run keeps it.
    Kept(KeptBullet),
}

/// The main bullets one input document carried, by utterance ordinal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GivenMainBullets {
    by_utterance: Vec<Given>,
    absence: AbsencePolicy,
}

impl GivenMainBullets {
    /// The slot for one ordinal. An ordinal past the input's utterances is a
    /// typed error: the working document has an utterance the input did not,
    /// and no given bullet can be attributed to it.
    fn slot(&self, utterance_idx: UtteranceIdx) -> Result<GivenSlot, KeptBulletError> {
        let given = self.by_utterance.get(utterance_idx.raw()).copied().ok_or(
            KeptBulletError::UtteranceOutsideInput {
                utterance_idx: utterance_idx.raw(),
                input_utterances: self.by_utterance.len(),
            },
        )?;
        Ok(match (given, self.absence) {
            (Given::Kept(kept), _) => GivenSlot::Kept(kept),
            (Given::Unbulleted, AbsencePolicy::Derive) => GivenSlot::Unbulleted,
            (Given::Unbulleted, AbsencePolicy::Keep) => GivenSlot::KeptAbsent,
        })
    }
}

/// Which main bullets a projection must leave exactly as the input gave them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MainBulletAuthority {
    /// No bullet is read-only: every bullet is a projection of its words.
    DeriveFromWords,
    /// The bullets the input carried are read-only (`keep`), and under
    /// `exact` so is the absence of one.
    Keep(GivenMainBullets),
}

/// What a phase running BEFORE [`MainBulletAuthority::impose`] may do to one
/// utterance's main bullet. `ReadOnly` covers a kept bullet and a kept
/// absence alike: either way the phase must not write one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum GivenMutability {
    /// The input's bullet, or its absence, is kept.
    ReadOnly,
    /// This bullet is the projection's to derive, widen or strip.
    Revisable,
}

/// What a phase after [`MainBulletAuthority::impose`] may do to one
/// utterance's LIVE main bullet.
///
/// Matched exhaustively by every phase that writes a bullet, so a phase added
/// later has to say what it does with a read-only bullet before it compiles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum BulletMutability {
    /// The input gave this bullet and the run keeps given bullets.
    ReadOnly(KeptBullet),
    /// This bullet is the projection's to derive, cut or strip.
    Revisable,
}

/// The input carried a bullet whose end precedes its start.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error(
    "--main-bullets keep: utterance {utterance_ordinal} (line {line_idx}) has a backward bullet \
     {start_ms}_{end_ms}; a backward bullet cannot be kept"
)]
pub struct BackwardGivenBullet {
    utterance_ordinal: usize,
    line_idx: usize,
    start_ms: u64,
    end_ms: u64,
}

/// A kept bullet could not be attributed or did not hold.
///
/// Each is an internal invariant failure, never bad input: it fails the file
/// rather than writing a transcript whose kept bullets moved.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum KeptBulletError {
    /// The working document has more utterances than the input did.
    #[error(
        "--main-bullets keep: utterance {utterance_idx} is not in the input, which has \
         {input_utterances} utterances"
    )]
    UtteranceOutsideInput {
        /// The working document's utterance ordinal.
        utterance_idx: usize,
        /// How many utterances the input had.
        input_utterances: usize,
    },
    /// The working document has fewer utterances than the input did.
    #[error(
        "--main-bullets keep: the input has {input_utterances} utterances but the output has \
         {working_utterances}"
    )]
    UtteranceMissing {
        /// How many utterances the input had.
        input_utterances: usize,
        /// How many the output has.
        working_utterances: usize,
    },
    /// A kept bullet is not what the input gave.
    #[error(
        "--main-bullets keep: utterance {utterance_idx} was given {given_start_ms}_{given_end_ms} \
         but now has {live}"
    )]
    Drifted {
        /// The utterance whose kept bullet moved.
        utterance_idx: usize,
        /// The given start.
        given_start_ms: u64,
        /// The given end.
        given_end_ms: u64,
        /// The live bullet, as `start_end`, or `no bullet`.
        live: String,
    },
    /// An utterance the input left without a bullet, under `exact`, has one.
    #[error(
        "--main-bullets exact: utterance {utterance_idx} was given no bullet but now has \
         {start_ms}_{end_ms}"
    )]
    GainedBullet {
        /// The utterance that gained a bullet.
        utterance_idx: usize,
        /// The live start.
        start_ms: u64,
        /// The live end.
        end_ms: u64,
    },
}

impl KeptBulletError {
    fn gained(utterance_idx: UtteranceIdx, live: &Bullet) -> Self {
        Self::GainedBullet {
            utterance_idx: utterance_idx.raw(),
            start_ms: live.timing.start_ms,
            end_ms: live.timing.end_ms,
        }
    }

    fn drifted(utterance_idx: UtteranceIdx, kept: KeptBullet, live: Option<&Bullet>) -> Self {
        Self::Drifted {
            utterance_idx: utterance_idx.raw(),
            given_start_ms: kept.start_ms(),
            given_end_ms: kept.end_ms(),
            live: match live {
                Some(bullet) => format!("{}_{}", bullet.timing.start_ms, bullet.timing.end_ms),
                None => "no bullet".to_string(),
            },
        }
    }
}

impl MainBulletAuthority {
    /// Bind the policy to the input document AS PARSED, before any pre-pass.
    ///
    /// The one place [`MainBulletPolicy`] is read for the projection. The
    /// default reads nothing and cannot fail; `keep` and `exact` classify
    /// every input bullet and refuse a backward one, and differ only in what
    /// an utterance WITHOUT a bullet becomes.
    pub fn bind(policy: MainBulletPolicy, input: &ChatFile) -> Result<Self, BackwardGivenBullet> {
        let absence = match policy {
            MainBulletPolicy::DeriveFromWords => return Ok(Self::DeriveFromWords),
            MainBulletPolicy::KeepGiven => AbsencePolicy::Derive,
            MainBulletPolicy::KeepExact => AbsencePolicy::Keep,
        };
        let by_utterance = utterances_indexed(input)
            .map(|(line_idx, utterance_idx, utterance)| {
                match utterance
                    .main
                    .content
                    .bullet
                    .as_ref()
                    .map(GivenMainBullet::classify)
                {
                    None => Ok(Given::Unbulleted),
                    Some(GivenMainBullet::Extent { start_ms, end_ms }) => {
                        Ok(Given::Kept(KeptBullet(Kept::Extent { start_ms, end_ms })))
                    }
                    Some(GivenMainBullet::Empty { at_ms }) => {
                        Ok(Given::Kept(KeptBullet(Kept::Empty { at_ms })))
                    }
                    Some(GivenMainBullet::Backward { start_ms, end_ms }) => {
                        Err(BackwardGivenBullet {
                            utterance_ordinal: utterance_idx.raw(),
                            line_idx,
                            start_ms,
                            end_ms,
                        })
                    }
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self::Keep(GivenMainBullets {
            by_utterance,
            absence,
        }))
    }

    /// What a phase running BEFORE [`Self::impose`] may do to a bullet. No
    /// live comparison: a pre-grouping pass may legitimately have widened it.
    pub(super) fn given_mutability(
        &self,
        utterance_idx: UtteranceIdx,
    ) -> Result<GivenMutability, KeptBulletError> {
        match self.slot(utterance_idx)? {
            None | Some(GivenSlot::Unbulleted) => Ok(GivenMutability::Revisable),
            Some(GivenSlot::Kept(_) | GivenSlot::KeptAbsent) => Ok(GivenMutability::ReadOnly),
        }
    }

    /// The input's slot for one utterance, or `None` when no bullet is kept
    /// at all (the default policy).
    fn slot(&self, utterance_idx: UtteranceIdx) -> Result<Option<GivenSlot>, KeptBulletError> {
        match self {
            Self::DeriveFromWords => Ok(None),
            Self::Keep(given) => given.slot(utterance_idx).map(Some),
        }
    }

    /// Put every kept bullet back exactly as given and clamp its words into it.
    ///
    /// The first step of `FaApplied::then_finalize`, on every route to
    /// finalization (a fresh injection, the all-`%wor` fast path, a partially
    /// reused utterance no group re-aligned, the incremental path). Words are
    /// clamped on BOTH tiers by [`clamp_words_within`]: a word crossing an edge
    /// is cut to it, a word wholly outside (or left with no extent) loses its
    /// timing. Clamping `%wor` in place is what keeps the contract when no
    /// `%wor` write is requested (`--nowor`, CA transcripts): the existing
    /// tier is never left with words outside its bullet. Utterances whose
    /// MAIN-tier words were cut are returned as [`KeptWordCuts`], for the
    /// write phase to regenerate `%wor` from; a cut on `%wor` alone is already
    /// in place and regenerating from an untimed main tier would erase it.
    ///
    /// Under [`Self::DeriveFromWords`] this touches nothing.
    pub(super) fn impose(
        &self,
        chat_file: &mut ChatFile,
    ) -> Result<Imposition<'_>, KeptBulletError> {
        let mut cuts = Vec::new();
        let mut records = Vec::new();
        if let Self::Keep(given) = self {
            for (line_idx, utterance_idx, utterance) in utterances_indexed_mut(chat_file) {
                let kept = match given.slot(utterance_idx)? {
                    GivenSlot::Unbulleted => continue,
                    GivenSlot::KeptAbsent => {
                        // No bullet, and no word timing a later reader could
                        // derive one from. An empty bound fits no word, so
                        // `clamp_words_within` untimes every word on both
                        // tiers, recording where the aligner had put each.
                        utterance.main.content.bullet = None;
                        let untimed = clamp_words_within(utterance, TimeSpan::new(0, 0));
                        if untimed.cut_main_tier() {
                            cuts.push(utterance_idx);
                        }
                        if let Some(reason) = absence_reason(&untimed) {
                            records.push(DecisionRecord::new_and_trace(
                                line_idx,
                                utterance.main.speaker.as_str().to_string(),
                                DecisionStrategy::Fa(FaStrategy::WordsUntimedForKeptAbsence),
                                reason,
                                false,
                            ));
                        }
                        continue;
                    }
                    GivenSlot::Kept(kept) => kept,
                };
                utterance.main.content.bullet = Some(kept.to_bullet());
                let clamped = clamp_words_within(utterance, kept.span());
                if clamped.cut_main_tier() {
                    cuts.push(utterance_idx);
                }
                if let Some((reason, needs_review)) = imposition_reason(kept, &clamped) {
                    records.push(DecisionRecord::new_and_trace(
                        line_idx,
                        utterance.main.speaker.as_str().to_string(),
                        DecisionStrategy::Fa(FaStrategy::WordsClampedToKeptBullet),
                        reason,
                        needs_review,
                    ));
                }
            }
        }
        Ok(Imposition {
            proof: ImposedBullets { authority: self },
            cuts: KeptWordCuts(cuts),
            records,
        })
    }

    /// Check, after every phase has run, that each kept bullet is exactly the
    /// given one and that the output has exactly the input's utterances.
    pub(super) fn verify_held(
        &self,
        chat_file: &ChatFile,
    ) -> Result<KeptBulletsHeld, KeptBulletError> {
        let given = match self {
            Self::DeriveFromWords => return Ok(KeptBulletsHeld(HeldBullets::NoneKept)),
            Self::Keep(given) => given,
        };
        let mut working_utterances = 0usize;
        let mut kept_count = 0usize;
        let mut kept_absent = 0usize;
        for (_, utterance_idx, utterance) in utterances_indexed(chat_file) {
            working_utterances += 1;
            match given.slot(utterance_idx)? {
                GivenSlot::Unbulleted => {}
                GivenSlot::KeptAbsent => match utterance.main.content.bullet.as_ref() {
                    None => kept_absent += 1,
                    Some(live) => return Err(KeptBulletError::gained(utterance_idx, live)),
                },
                GivenSlot::Kept(kept) => {
                    let live = utterance.main.content.bullet.as_ref();
                    match kept.is_live(live) {
                        true => kept_count += 1,
                        false => return Err(KeptBulletError::drifted(utterance_idx, kept, live)),
                    }
                }
            }
        }
        match working_utterances == given.by_utterance.len() {
            true => Ok(KeptBulletsHeld(HeldBullets::AllHeld {
                kept: kept_count,
                kept_absent,
            })),
            false => Err(KeptBulletError::UtteranceMissing {
                input_utterances: given.by_utterance.len(),
                working_utterances,
            }),
        }
    }
}

/// The decision reason for one kept utterance, or `None` when nothing needs
/// saying (a positive-extent bullet whose words already fitted).
fn imposition_reason(kept: KeptBullet, clamped: &ClampedWordCounts) -> Option<(String, bool)> {
    let mut reason = String::new();
    let needs_review = match kept.0 {
        Kept::Extent { start_ms, end_ms } => {
            if clamped.trimmed() == 0 && clamped.dropped.is_empty() {
                return None;
            }
            let _ = write!(reason, "kept_main_bullet={start_ms}_{end_ms}");
            !clamped.dropped.is_empty()
        }
        // An empty bullet is always worth a record: the input asserted an
        // utterance that took no time, and every word of it is now untimed.
        Kept::Empty { at_ms } => {
            let _ = write!(reason, "kept_empty_main_bullet={at_ms}_{at_ms}");
            true
        }
    };
    let _ = write!(
        reason,
        " words_trimmed={} words_dropped={} ",
        clamped.trimmed(),
        clamped.dropped.len()
    );
    write_dropped(&mut reason, clamped);
    let _ = write!(reason, " cause=word_outside_kept_bullet");
    Some((reason, needs_review))
}

/// The decision reason for one kept absence whose words the aligner had
/// timed, or `None` when none was timed.
fn absence_reason(untimed: &ClampedWordCounts) -> Option<String> {
    if untimed.dropped.is_empty() {
        return None;
    }
    let mut reason = format!(
        "kept_absent_main_bullet words_untimed={} ",
        untimed.dropped.len()
    );
    write_dropped(&mut reason, untimed);
    let _ = write!(reason, " cause=utterance_given_no_bullet");
    Some(reason)
}

/// `dropped=[tier:word:start_end,...]`, the one spelling both reasons use for
/// where each word had been placed before its timing was removed.
fn write_dropped(reason: &mut String, clamped: &ClampedWordCounts) {
    reason.push_str("dropped=[");
    for (i, dropped) in clamped.dropped.iter().enumerate() {
        let tier = match dropped.tier {
            WordTier::MainTier => "main",
            WordTier::Wor => "wor",
        };
        let separator = if i == 0 { "" } else { "," };
        let _ = write!(
            reason,
            "{separator}{tier}:{}:{}_{}",
            dropped.word_index, dropped.measured.start_ms, dropped.measured.end_ms
        );
    }
    reason.push(']');
}

/// What [`MainBulletAuthority::impose`] did, consumed by `then_finalize`.
#[must_use = "the proof gates repair and monotonicity; the cuts must reach the %wor plan"]
pub(super) struct Imposition<'a> {
    proof: ImposedBullets<'a>,
    cuts: KeptWordCuts,
    records: Vec<DecisionRecord>,
}

impl<'a> Imposition<'a> {
    /// The proof, the `%wor` rewrites, and the decision records.
    pub(super) fn into_parts(self) -> (ImposedBullets<'a>, KeptWordCuts, Vec<DecisionRecord>) {
        (self.proof, self.cuts, self.records)
    }
}

/// Utterances whose main-tier words `impose` cut, so their `%wor` must be
/// regenerated by the write phase (when one is requested).
#[must_use = "a kept utterance whose words were cut must reach the %wor plan"]
pub(super) struct KeptWordCuts(Vec<UtteranceIdx>);

impl KeptWordCuts {
    /// Hand the utterances to the `%wor` plan.
    pub(super) fn into_utterances(self) -> Vec<UtteranceIdx> {
        self.0
    }
}

/// Proof that [`MainBulletAuthority::impose`] ran: every kept bullet was put
/// back and its words fitted. Required by bullet repair and monotonicity.
///
/// The private field is what makes it a proof: the only constructors are
/// `impose` and, for the test-only derive entry points, [`Self::deriving`].
#[derive(Clone, Copy)]
pub(super) struct ImposedBullets<'a> {
    authority: &'a MainBulletAuthority,
}

impl ImposedBullets<'_> {
    /// What a phase may do to this utterance's bullet, checked against the
    /// LIVE bullet: a kept bullet that is no longer the given one is an error
    /// here, at the first phase to see it.
    ///
    /// Only an utterance with a live bullet is asked about, so a kept absence
    /// here has gained one, which is [`KeptBulletError::GainedBullet`].
    pub(super) fn mutability(
        &self,
        utterance_idx: UtteranceIdx,
        live: &Bullet,
    ) -> Result<BulletMutability, KeptBulletError> {
        match self.authority.slot(utterance_idx)? {
            None | Some(GivenSlot::Unbulleted) => Ok(BulletMutability::Revisable),
            Some(GivenSlot::KeptAbsent) => Err(KeptBulletError::gained(utterance_idx, live)),
            Some(GivenSlot::Kept(kept)) => match kept.is_live(Some(live)) {
                true => Ok(BulletMutability::ReadOnly(kept)),
                false => Err(KeptBulletError::drifted(utterance_idx, kept, Some(live))),
            },
        }
    }
}

#[cfg(test)]
static DERIVE_FROM_WORDS: MainBulletAuthority = MainBulletAuthority::DeriveFromWords;

#[cfg(test)]
impl ImposedBullets<'static> {
    /// The proof under the default policy, for the test-only derive entry
    /// points. Imposing nothing is all `impose` does under that policy.
    pub(super) fn deriving() -> Self {
        Self {
            authority: &DERIVE_FROM_WORDS,
        }
    }
}

/// Proof, carried by `FaFinalized`, that every kept bullet held.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeptBulletsHeld(HeldBullets);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HeldBullets {
    /// The run kept no bullets (the default policy).
    NoneKept,
    /// Every given bullet, this many, is exactly the one written, and every
    /// kept absence, this many, is still without one.
    AllHeld { kept: usize, kept_absent: usize },
}

impl KeptBulletsHeld {
    /// How many given bullets were verified unchanged (0 under the default).
    pub fn kept(self) -> usize {
        match self.0 {
            HeldBullets::NoneKept => 0,
            HeldBullets::AllHeld { kept, .. } => kept,
        }
    }

    /// How many kept absences were verified still without a bullet (0 unless
    /// `exact`).
    pub fn kept_absent(self) -> usize {
        match self.0 {
            HeldBullets::NoneKept => 0,
            HeldBullets::AllHeld { kept_absent, .. } => kept_absent,
        }
    }
}

/// A projection policy together with the bullets its main-bullet choice binds.
///
/// Every projection entry point takes this, so a phase cannot consult a
/// policy without the bullets it refers to. Not `Clone`: one binding is spent
/// by one finalization.
#[derive(Debug)]
pub struct FaProjection {
    policy: FaProjectionPolicy,
    main_bullets: MainBulletAuthority,
}

impl FaProjection {
    /// Pair a projection policy with the authority bound at the parse.
    pub fn new(policy: FaProjectionPolicy, main_bullets: MainBulletAuthority) -> Self {
        Self {
            policy,
            main_bullets,
        }
    }

    /// The bound bullets, for a pre-grouping pass that must label what it
    /// does to a kept bullet (narrow-bullet rescue).
    pub(crate) fn main_bullets(&self) -> &MainBulletAuthority {
        &self.main_bullets
    }

    /// Split into the policy and the bullet authority, for `FaApplied`.
    pub(super) fn into_parts(self) -> (FaProjectionPolicy, MainBulletAuthority) {
        (self.policy, self.main_bullets)
    }
}
