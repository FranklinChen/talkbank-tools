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

enum StartOrderPosition<'utterance> {
    Accepted,
    Regressing(StartRegression<'utterance>),
    Yielding(StartYield<'utterance>),
    Ahead(StartAhead<'utterance>),
}

/// A revisable start AFTER a later utterance's accepted start: an outlier the
/// chain of starts left out (a mistyped `600000` for `6000`). Stripping it is
/// what keeps every later utterance's timing; accepting it, as the greedy rule
/// did, made it the floor every later start fell below.
struct StartAhead<'utterance> {
    target: StartOrderTarget<'utterance>,
    following: Arc<StartOrderEntry>,
}

struct ResolvedStartAhead {
    following: Arc<StartOrderEntry>,
    timing: StrippedTiming,
}

impl StartAhead<'_> {
    fn resolve(self) -> ResolvedStartAhead {
        ResolvedStartAhead {
            following: self.following,
            timing: strip_utterance_timing(self.target.utterance),
        }
    }
}

/// Only the plan can bind a revisable target to a conflicting kept
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

/// Plan one segment: the revisable starts between two kept starts, in file
/// order, with the last valid kept start before them as `floor` and the kept
/// start that ends the segment as `ceiling`. Each target is consumed into its
/// own position in one file-order pass; the per-position facts it is judged
/// by (its chain role, the next chain start) are derived from the same
/// segment, one entry per target, and zipped with it in that pass.
///
/// A start above the ceiling yields to it. The unmarked starts within the
/// bounds that are kept form a non-decreasing chain chosen so that the most
/// timings survive ([`weighted_chain`]): one outlier start (a typed `600000`
/// for `6000`) is the one stripped, where the greedy rule made it the floor
/// and stripped every later utterance, and two `+<` backchannels never
/// outvote the turn they overlap. Every other start is then judged in file
/// order against the greatest start accepted before it (the floor at first):
/// below it, it regresses (a `+<` start is raised, an unmarked one
/// stripped); above the next chain start, it is ahead and stripped;
/// otherwise it is accepted.
fn plan_segment<'utterance>(
    segment: Vec<StartOrderTarget<'utterance>>,
    floor: Option<&Arc<StartOrderEntry>>,
    ceiling: Option<&Arc<StartOrderEntry>>,
) -> Vec<StartOrderPosition<'utterance>> {
    let inputs: Vec<ChainInput> = segment
        .iter()
        .map(|target| ChainInput::of(&target.facts, floor, ceiling))
        .collect();
    let roles = weighted_chain(&inputs, floor.map(|floor| floor.start_ms));
    // The next chain start after each position.
    let mut following_of: Vec<Option<Arc<StartOrderEntry>>> = vec![None; segment.len()];
    let mut following: Option<Arc<StartOrderEntry>> = None;
    for index in (0..segment.len()).rev() {
        following_of[index] = following.clone();
        match roles[index] {
            ChainRole::Member => following = Some(segment[index].facts.clone()),
            ChainRole::NotMember => {}
        }
    }
    // In file order, against the greatest start accepted so far (the floor,
    // a chain start, or an accepted `+<` start), as the greedy rule judged
    // every start: so `+<` starts keep their order among themselves.
    let mut previous: Option<Arc<StartOrderEntry>> = floor.cloned();
    let mut positions = Vec::with_capacity(segment.len());
    for ((target, role), following) in segment.into_iter().zip(roles).zip(following_of) {
        let position = match Bound::of(&target.facts, floor, ceiling) {
            Bound::AboveCeiling(kept) => StartOrderPosition::Yielding(StartYield { target, kept }),
            // Below the floor is judged like any other start, against the
            // greatest start accepted before it, which is never below the
            // floor: raising a `+<` start to the floor alone wrote a start
            // below an accepted one (2026-11-12 review).
            Bound::BelowFloor | Bound::Within => {
                match Position::of(&target.facts, role, previous.as_ref(), following.as_ref()) {
                    // An accepted start keeps its bullet: the target, its only
                    // mutation capability, is dropped here unused.
                    Position::Chained | Position::Between => {
                        previous = Some(target.facts.clone());
                        StartOrderPosition::Accepted
                    }
                    Position::BelowPrevious(previous) => {
                        StartOrderPosition::Regressing(StartRegression {
                            entry: target,
                            previous,
                        })
                    }
                    Position::AboveFollowing(following) => {
                        StartOrderPosition::Ahead(StartAhead { target, following })
                    }
                }
            }
        };
        positions.push(position);
    }
    positions
}

/// Where a start stands against its segment's kept bounds.
enum Bound {
    /// Below the last valid kept start before the segment.
    BelowFloor,
    /// Above the kept start that ends the segment.
    AboveCeiling(Arc<StartOrderEntry>),
    /// Between them.
    Within,
}

impl Bound {
    fn of(
        entry: &StartOrderEntry,
        floor: Option<&Arc<StartOrderEntry>>,
        ceiling: Option<&Arc<StartOrderEntry>>,
    ) -> Self {
        if let Some(floor) = floor
            && entry.start_ms < floor.start_ms
        {
            return Self::BelowFloor;
        }
        if let Some(kept) = ceiling
            && entry.start_ms > kept.start_ms
        {
            return Self::AboveCeiling(kept.clone());
        }
        Self::Within
    }
}

/// Where a start within its kept bounds stands against its accepted
/// neighbours.
enum Position {
    /// In the chain of unmarked starts kept in order.
    Chained,
    /// Below the greatest start accepted before it.
    BelowPrevious(Arc<StartOrderEntry>),
    /// Above the next chain start after it.
    AboveFollowing(Arc<StartOrderEntry>),
    /// In order with both: a `+<` start between its neighbours, or an
    /// unmarked start the chain left out though it is in order with both
    /// (leaving it out kept more `+<` timings): either keeps its timing.
    Between,
}

impl Position {
    fn of(
        entry: &StartOrderEntry,
        role: ChainRole,
        previous: Option<&Arc<StartOrderEntry>>,
        following: Option<&Arc<StartOrderEntry>>,
    ) -> Self {
        match role {
            ChainRole::Member => Self::Chained,
            ChainRole::NotMember => Self::against(entry, previous, following),
        }
    }

    /// A start outside the chain, against the greatest start accepted before
    /// it and the next chain start after it.
    fn against(
        entry: &StartOrderEntry,
        previous: Option<&Arc<StartOrderEntry>>,
        following: Option<&Arc<StartOrderEntry>>,
    ) -> Self {
        if let Some(previous) = previous
            && entry.start_ms < previous.start_ms
        {
            return Self::BelowPrevious(previous.clone());
        }
        if let Some(next) = following
            && entry.start_ms > next.start_ms
        {
            return Self::AboveFollowing(next.clone());
        }
        Self::Between
    }
}

/// What one start contributes to the choice of chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChainInput {
    /// An unmarked start within the kept bounds: a possible chain member. Left
    /// out, it loses its timing.
    Member(u64),
    /// A `+<` start within the kept bounds. Never a member. Judged against
    /// the start accepted before it, it keeps its timing when it is at or
    /// above that start (accepted) or when a raise to that start leaves it
    /// something (see [`StartRegression::resolve`]), and only while it is not
    /// above the next member (there it is ahead and stripped).
    Overlap {
        /// Its start.
        start: u64,
        /// The greatest start it can be judged against and keep its timing.
        keeps_against: KeepThreshold,
    },
    /// Outside the kept bounds: judged against them, not by the chain.
    Outside,
}

impl ChainInput {
    fn of(
        entry: &StartOrderEntry,
        floor: Option<&Arc<StartOrderEntry>>,
        ceiling: Option<&Arc<StartOrderEntry>>,
    ) -> Self {
        match (Bound::of(entry, floor, ceiling), entry.overlap) {
            (Bound::Within, TranscribedOverlap::Unmarked) => Self::Member(entry.start_ms),
            (Bound::Within, TranscribedOverlap::PrecedesMarked(reach)) => Self::Overlap {
                start: entry.start_ms,
                keeps_against: reach.keeps_against(entry.start_ms),
            },
            (Bound::BelowFloor | Bound::AboveCeiling(_), _) => Self::Outside,
        }
    }
}

/// Whether a start is a member of the chosen chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChainRole {
    Member,
    NotMember,
}

/// Counts of `Overlap` thresholds inserted so far, queried by "how many keep
/// their timing against this start": a Fenwick tree over the segment's
/// thresholds, so the chain search stays O(n² log n).
struct KeptCounts {
    thresholds: Vec<KeepThreshold>,
    tree: Vec<usize>,
    inserted: usize,
}

impl KeptCounts {
    fn over(inputs: &[ChainInput]) -> Self {
        let mut thresholds: Vec<KeepThreshold> = inputs
            .iter()
            .filter_map(|input| match input {
                ChainInput::Overlap { keeps_against, .. } => Some(*keeps_against),
                ChainInput::Member(_) | ChainInput::Outside => None,
            })
            .collect();
        thresholds.sort_unstable();
        thresholds.dedup();
        let tree = vec![0; thresholds.len() + 1];
        Self {
            thresholds,
            tree,
            inserted: 0,
        }
    }

    fn clear(&mut self) {
        self.tree.fill(0);
        self.inserted = 0;
    }

    /// Count one `Overlap` start. Its threshold is one `over` collected from
    /// the same inputs, so its rank is its exact position.
    fn insert(&mut self, keeps_against: KeepThreshold) {
        let mut at = self.thresholds.partition_point(|&t| t < keeps_against) + 1;
        while at < self.tree.len() {
            self.tree[at] += 1;
            at += at.isolate_lowest_one();
        }
        self.inserted += 1;
    }

    /// How many inserted `Overlap` starts keep their timing judged against
    /// `start`: those whose threshold is at least `start`.
    fn keeping_against(&self, start: u64) -> usize {
        let mut at = self
            .thresholds
            .partition_point(|&t| !t.keeps_against_start(start));
        let mut below = 0;
        while at > 0 {
            below += self.tree[at];
            at -= at.isolate_lowest_one();
        }
        self.inserted - below
    }

    /// How many inserted `Overlap` starts keep their timing against the
    /// floor; every one, with no floor.
    fn keeping_against_floor(&self, floor: Option<u64>) -> usize {
        match floor {
            Some(floor) => self.keeping_against(floor),
            None => self.inserted,
        }
    }
}

/// A chain member's best chain so far: what it keeps up to and including
/// itself, its start, and the member before it.
#[derive(Debug, Clone, Copy)]
struct Link {
    start: u64,
    through: usize,
    from: Option<usize>,
}

/// One position of the chain search, filled in file order, so every slot
/// before a member is complete when the member is searched: a member always
/// carries its link.
#[derive(Debug, Clone, Copy)]
enum ChainSlot {
    Member(Link),
    Overlap {
        start: u64,
        keeps_against: KeepThreshold,
    },
    Outside,
}

/// Which starts are chain members: the non-decreasing chain of `Member`
/// starts that keeps the most timings, counting its members and every
/// `Overlap` start that keeps its timing judged against the member before it
/// (the floor before the first) and is not above the member after it.
/// Counting only members let two backchannels below one turn outvote it;
/// counting every `+<` start not above the next member let a chain be chosen
/// whose raises strip what another chain would have kept (2026-11-12
/// reviews).
///
/// Two residuals the score does not model, both because the file-order pass
/// judges each start against the greatest start ACCEPTED so far, not the
/// member before it: an accepted `+<` start between two members, and an
/// unmarked start the chain left out that the pass accepts as in order
/// (`Position::Between`), each raise the start later `+<` starts are judged
/// against, so the score can count one that the pass then strips. Neither
/// makes the result worse than the greedy rule it replaced. Ties: the empty
/// chain wins a tie with any chain, and of chains tied with each other the
/// one through the earliest utterances, so document order decides where
/// nothing measured does (two swapped starts keep the first). O(n² log n) in
/// the segment's starts, and with no kept bullet the segment is the file.
fn weighted_chain(inputs: &[ChainInput], floor: Option<u64>) -> Vec<ChainRole> {
    let mut counts = KeptCounts::over(inputs);
    let mut slots: Vec<ChainSlot> = Vec::with_capacity(inputs.len());
    for input in inputs {
        let slot = match *input {
            ChainInput::Overlap {
                start,
                keeps_against,
            } => ChainSlot::Overlap {
                start,
                keeps_against,
            },
            ChainInput::Outside => ChainSlot::Outside,
            ChainInput::Member(start) => {
                // Walking back over the slots before it, `counts` holds the
                // `Overlap` starts strictly between the candidate predecessor
                // and this member that are not above it.
                counts.clear();
                // The best predecessor so far, as (score, position).
                let mut choice: Option<(usize, usize)> = None;
                for (i, slot) in slots.iter().enumerate().rev() {
                    match *slot {
                        ChainSlot::Member(link) if link.start <= start => {
                            let score = link.through + counts.keeping_against(link.start);
                            // `>=` walking backwards keeps the earliest
                            // predecessor of equal score.
                            if choice.is_none_or(|(best_score, _)| score >= best_score) {
                                choice = Some((score, i));
                            }
                        }
                        ChainSlot::Overlap {
                            start: overlap,
                            keeps_against,
                        } if overlap <= start => counts.insert(keeps_against),
                        ChainSlot::Member(_) | ChainSlot::Overlap { .. } | ChainSlot::Outside => {}
                    }
                }
                // As the first member: the `Overlap` starts before it, judged
                // against the floor; a predecessor only when it keeps at
                // least as much.
                let first = counts.keeping_against_floor(floor);
                let (through, from) = match choice.filter(|(score, _)| *score >= first) {
                    Some((score, i)) => (score, Some(i)),
                    None => (first, None),
                };
                ChainSlot::Member(Link {
                    start,
                    through: through + 1,
                    from,
                })
            }
        };
        slots.push(slot);
    }
    // One reverse sweep: each member's total adds the `Overlap` starts after
    // it judged against it; at the end `counts` holds every `Overlap` start,
    // which is what the empty chain keeps against the floor. `>=` walking
    // backwards keeps the earliest member of equal total; the empty chain
    // wins a tie with any chain.
    counts.clear();
    // The best last member, as (total, position, link).
    let mut last: Option<(usize, usize, Link)> = None;
    for (j, slot) in slots.iter().enumerate().rev() {
        match *slot {
            ChainSlot::Member(link) => {
                let total = link.through + counts.keeping_against(link.start);
                if last.is_none_or(|(best, _, _)| total >= best) {
                    last = Some((total, j, link));
                }
            }
            ChainSlot::Overlap { keeps_against, .. } => counts.insert(keeps_against),
            ChainSlot::Outside => {}
        }
    }
    let empty = counts.keeping_against_floor(floor);
    let mut roles = vec![ChainRole::NotMember; inputs.len()];
    let mut cursor = last
        .filter(|(total, _, _)| *total > empty)
        .map(|(_, j, link)| (j, link));
    while let Some((j, link)) = cursor {
        roles[j] = ChainRole::Member;
        // A link's `from` names a member slot by construction.
        cursor = link.from.and_then(|i| match slots[i] {
            ChainSlot::Member(previous) => Some((i, previous)),
            ChainSlot::Overlap { .. } | ChainSlot::Outside => None,
        });
    }
    roles
}

#[cfg(test)]
mod chain_tests {
    use super::{ChainInput, ChainRole, RaiseReach, weighted_chain};
    use ChainInput::{Member, Outside};
    use ChainRole::{Member as In, NotMember as Out};

    /// A `+<` start whose timing reaches `reach`, through the one rule.
    fn overlap(start: u64, reach: u64) -> ChainInput {
        ChainInput::Overlap {
            start,
            keeps_against: RaiseReach::reaching(reach).keeps_against(start),
        }
    }

    /// One outlier start (a mistyped 600000 for 6000) is the one left out;
    /// every later start is kept.
    #[test]
    fn an_outlier_start_is_the_one_left_out() {
        let starts = [1_000, 600_000, 7_000, 8_000, 9_000].map(Member);
        assert_eq!(weighted_chain(&starts, None), [In, Out, In, In, In]);
    }

    /// Where two chains tie, the earlier utterance is kept.
    #[test]
    fn a_tie_keeps_document_order() {
        assert_eq!(
            weighted_chain(&[Member(5_000), Member(2_000)], None),
            [In, Out]
        );
        // Equal starts are in order.
        assert_eq!(
            weighted_chain(&[Member(3_000), Member(3_000), Member(1_000)], None),
            [In, In, Out]
        );
    }

    /// A start outside the kept bounds never joins.
    #[test]
    fn outside_starts_never_join() {
        assert_eq!(
            weighted_chain(&[Member(1_000), Outside, Member(3_000)], None),
            [In, Out, In]
        );
    }

    /// Two backchannels below a turn do not outvote it: raised to its start
    /// they still keep their timing (they reach past it).
    #[test]
    fn backchannels_below_a_turn_never_outvote_it() {
        let starts = [
            Member(1_000),
            Member(3_000),
            overlap(2_500, 3_500),
            overlap(2_700, 3_600),
        ];
        assert_eq!(weighted_chain(&starts, None), [In, In, Out, Out]);
    }

    /// One mistyped turn (1200 for 2200) does not strip the backchannels above
    /// it: leaving the turn out keeps four timings, keeping it would keep two.
    #[test]
    fn one_mistyped_turn_does_not_strip_the_backchannels_above_it() {
        let starts = [
            Member(1_000),
            overlap(1_100, 1_300),
            overlap(1_200, 1_400),
            overlap(1_300, 1_500),
            overlap(1_400, 1_600),
            Member(1_200),
            Member(2_000),
        ];
        assert_eq!(
            weighted_chain(&starts, None),
            [In, Out, Out, Out, Out, Out, In]
        );
    }

    /// A backchannel that a raise would strip is not counted as kept: the
    /// outlier turn A (3000) is left out, so the backchannel at 2000, which
    /// ends at 2100, stays in order and keeps its timing (2026-11-12 review,
    /// round nine: counting it chose {A, C} and kept two timings, not three).
    #[test]
    fn a_backchannel_a_raise_would_strip_is_not_counted_as_kept() {
        let starts = [
            Member(3_000),
            overlap(2_000, 2_100),
            overlap(2_060, 2_070),
            Member(2_050),
            Member(5_000),
        ];
        assert_eq!(weighted_chain(&starts, None), [Out, Out, Out, In, In]);
    }
}

/// Whether the transcriber marked an utterance as overlapping the utterance
/// before it. Read from the parsed model's linkers, where Chatter puts the
/// `+<` ("lazy overlap precedes") marker; never a flag a caller sets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TranscribedOverlap {
    /// The main tier begins with `+<`: this turn starts while the previous
    /// turn is still going. It may be raised to the start before it, which
    /// keeps its timing only within its reach.
    PrecedesMarked(RaiseReach),
    /// No such marker: the transcriber placed this turn after the previous.
    Unmarked,
}

impl TranscribedOverlap {
    fn of(utterance: &Utterance, end_ms: u64) -> Self {
        match overlaps_previous_utterance(utterance) {
            true => Self::PrecedesMarked(RaiseReach::of(utterance, end_ms)),
            false => Self::Unmarked,
        }
    }
}

use reach::{KeepThreshold, RaiseReach};

/// The raise rule, in a module of its own so that its two values can be
/// made only by their constructors: nothing in the rest of this file can
/// write a reach or a threshold that the rule did not compute.
mod reach {
    use super::{Utterance, furthest_word_timing_end};

    /// How far a `+<` utterance's timing reaches: the earlier of its bullet's
    /// end and the latest end of any of its timed words (the bullet's end
    /// when no word is timed). The one statement of the raise rule, read both
    /// by the raise ([`super::StartRegression::resolve`]) and by the chain
    /// search that predicts it ([`super::ChainInput`]), so the two cannot
    /// drift apart.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(super) struct RaiseReach(u64);

    impl RaiseReach {
        pub(super) fn of(utterance: &Utterance, end_ms: u64) -> Self {
            Self(match furthest_word_timing_end(utterance) {
                Some(word_end) => end_ms.min(word_end),
                None => end_ms,
            })
        }

        /// A reach given directly, for tests of the chain search.
        #[cfg(test)]
        pub(super) fn reaching(end_ms: u64) -> Self {
            Self(end_ms)
        }

        /// Whether raising the start to `new_start` leaves the utterance any
        /// extent and, when a word is timed, a timed word ending after it.
        pub(super) fn survives_raise_to(self, new_start: u64) -> bool {
            new_start < self.0
        }

        /// The greatest start an utterance starting at `start` can be judged
        /// against and keep its timing: at or below its own start it is
        /// accepted as it is, and below its reach a raise keeps something.
        pub(super) fn keeps_against(self, start: u64) -> KeepThreshold {
            KeepThreshold(start.max(self.0.saturating_sub(1)))
        }
    }

    /// The greatest start a `+<` utterance can be judged against and keep
    /// its timing; made only by [`RaiseReach::keeps_against`].
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    pub(super) struct KeepThreshold(u64);

    impl KeepThreshold {
        /// Whether an utterance with this threshold keeps its timing judged
        /// against `start`.
        pub(super) fn keeps_against_start(self, start: u64) -> bool {
            start <= self.0
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
/// Minted only by [`plan_segment`], which owns the comparison, so holding one
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
            TranscribedOverlap::PrecedesMarked(reach) => {
                // A raise keeps the bullet only if it still has extent past
                // the new start and, when any word is timed, at least one
                // timed word ends after it ([`RaiseReach`]). An utterance with
                // a bullet and no timed word at all is raised: the raise
                // loses nothing it has.
                if !reach.survives_raise_to(new_start) {
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
/// Kept starts are a FIXED skeleton; the revisable starts between two of
/// them are planned together ([`plan_segment`]). The ones kept form the
/// chain of in-order starts that keeps the most timings ([`weighted_chain`]);
/// each other start is resolved in file order against the greatest start
/// accepted before it: below it, a start regression ([`StartRegression`]
/// raises an utterance the transcriber marked `+<` and strips any other);
/// above the next chain start, ahead and stripped; above the next kept
/// start, yielded to it and stripped. A kept start that regresses below an
/// earlier kept start cannot be fixed without changing a kept bullet, so it
/// is left and recorded as unresolved; it neither closes nor bounds the open
/// segment, which runs on to the next valid kept start. With no kept bullet
/// the whole file is one segment, and a mistyped start costs only its own
/// timing, where the greedy rule this replaced (strip every start below the
/// greatest preceding accepted start) made it the floor for everything after.
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
            overlap: TranscribedOverlap::of(utterance, bullet.timing.end_ms),
        });
        entries.push(match mutability {
            BulletMutability::ReadOnly(_) => StartOrderSource::Kept(facts),
            BulletMutability::Revisable => {
                StartOrderSource::Revisable(StartOrderTarget { facts, utterance })
            }
        });
    }

    // Revisable starts are planned a segment at a time (those between two
    // kept starts share a floor and a ceiling), from the facts alone, then
    // resolved in file order. Kept bullets are never changed here.
    let mut floor: Option<Arc<StartOrderEntry>> = None;
    let mut segment: Vec<StartOrderTarget<'_>> = Vec::new();
    for source in entries {
        match source {
            StartOrderSource::Kept(entry) => {
                match floor
                    .as_ref()
                    .filter(|previous| entry.start_ms < previous.start_ms)
                {
                    // A kept start below an earlier kept start is recorded and
                    // left. It neither ends the open segment nor bounds it:
                    // the segment runs on to the next valid kept start, as
                    // though this one had been stripped. Closing the segment
                    // on it stripped every revisable start before it for a
                    // violation that was its own.
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
                        resolve_segment(
                            std::mem::take(&mut segment),
                            floor.as_ref(),
                            Some(&entry),
                            decisions,
                            effects,
                        );
                        floor = Some(entry);
                    }
                }
            }
            StartOrderSource::Revisable(target) => segment.push(target),
        }
    }
    resolve_segment(segment, floor.as_ref(), None, decisions, effects);
    Ok(())
}

/// Plan one segment's revisable starts together, then resolve each in file
/// order, recording what was done.
fn resolve_segment(
    segment: Vec<StartOrderTarget<'_>>,
    floor: Option<&Arc<StartOrderEntry>>,
    ceiling: Option<&Arc<StartOrderEntry>>,
    decisions: &mut Vec<batchalign_transform::decisions::DecisionRecord>,
    effects: &mut Vec<MonotonicityEffect>,
) {
    use batchalign_transform::decisions::MonotonicityStrategy;

    for position in plan_segment(segment, floor, ceiling) {
        match position {
            StartOrderPosition::Regressing(regression) => {
                let entry = regression.entry.facts.clone();
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
                    }
                }
            }
            StartOrderPosition::Yielding(yielding) => {
                let entry = yielding.target.facts.clone();
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
            StartOrderPosition::Ahead(ahead) => {
                let entry = ahead.target.facts.clone();
                let ResolvedStartAhead {
                    following,
                    timing: StrippedTiming { words_dropped },
                } = ahead.resolve();
                decisions.push(start_order_record(
                    &entry,
                    MonotonicityStrategy::TimingStripped,
                    format!(
                        "ahead_of_following_start start_ms={} following_start_ms={} \
                                 following_line_idx={} following_speaker={}",
                        entry.start_ms, following.start_ms, following.line_idx, following.speaker
                    ),
                ));
                effects.push(MonotonicityEffect::StartAheadStripped {
                    line_idx: entry.line_idx,
                    utterance_idx: entry.utterance_idx,
                    speaker: entry.speaker.clone(),
                    start_ms: entry.start_ms,
                    following_start_ms: following.start_ms,
                    following_line_idx: following.line_idx,
                    following_utterance_idx: following.utterance_idx,
                    following_speaker: following.speaker.clone(),
                    words_dropped,
                });
            }
            StartOrderPosition::Accepted => {}
        }
    }
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
