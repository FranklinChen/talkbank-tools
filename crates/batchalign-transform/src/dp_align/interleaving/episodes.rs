//! Joint bounded adjacent-turn composition. Every complete source-DAG path
//! consumes every source word once, through singleton turns or disjoint episodes
//! of one turn and a following run of another speaker. Reference words are consumed once.
//! Independent local proposals cannot reserve the same ASR word twice here.
//!
//! Reference-layer checkpoints permit exact forward/backward proof without
//! keeping a recording-wide source-state × reference matrix in memory.

use super::{Alignable, FuzzyComparison, MatchMode, PreparedFuzzyWord, WordEvidence};
use crate::dp_align::LiteralMatchMode;
use crate::dp_align::control::{Uninterrupted, WorkControl, uninterrupted};

mod graph;
use graph::{CompletedGraphObservation, Edge, Graph, PreparedGraph};
mod band;
use band::{Band, BandAlternative, ForwardScores};
mod packed;
pub use packed::{PackedSearchCost, PackedSearchStage, PackedSearchStatus, PackedSearchWork};
mod projection;

const WORK_CELLS: usize = 64_000_000;
const SCORE_CELLS: usize = 2_097_152;
const CANDIDATES: usize = 1_048_576;

/// Borrowed speaker identity and its turn's ordered lexical census.
#[derive(Clone, Copy)]
pub struct SpeakerTurn<'source> {
    speaker: &'source str,
    words: &'source [String],
    continuation: Continuation,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Continuation {
    Separate,
    OverlappingSpeakerRun,
}

impl<'source> SpeakerTurn<'source> {
    /// Bind one source turn without reconstructing text from a wire artifact.
    pub fn new(speaker: &'source str, words: &'source [String]) -> Self {
        Self {
            speaker,
            words,
            continuation: Continuation::Separate,
        }
    }

    /// Declare source-observed overlap continuity for extending the following
    /// speaker's chain. This is an order-model input, not acoustic proof.
    pub fn overlapping_continuation(speaker: &'source str, words: &'source [String]) -> Self {
        Self {
            speaker,
            words,
            continuation: Continuation::OverlappingSpeakerRun,
        }
    }
}

/// Which admission stage exceeded the work budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocalInterleavingWorkPhase {
    /// Checked graph-size arithmetic overflowed before graph construction.
    GraphShape,
    /// Constructing the canonical-order feasible score is too expensive.
    FeasibleWitness,
    /// The proven score bound still admits too many source/reference cells.
    BandedSearch,
    /// Complete packed graph passes, reconstruction and edge observation.
    PackedSearch,
}

impl LocalInterleavingWorkPhase {
    /// Stable diagnostic spelling, not a correspondence certificate.
    pub fn name(self) -> &'static str {
        match self {
            Self::GraphShape => "graph_shape",
            Self::FeasibleWitness => "feasible_witness",
            Self::BandedSearch => "banded_search",
            Self::PackedSearch => "packed_search",
        }
    }
}

/// Why no analysis was produced: a named finite budget, or an internal
/// inconsistency of the search. Never an assertion of absence or ambiguity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocalInterleavingRefusal {
    /// A named finite budget prevented completion.
    Budget(LocalInterleavingBudget),
    /// Not a budget: the search contradicted what its own admission proved,
    /// and refused rather than report a wrong optimum.
    Inconsistent(LocalInterleavingInconsistency),
}

/// Bounded analysis refusal: one named finite budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocalInterleavingBudget {
    /// Feasible-witness work plus admitted source-DAG/reference cells exceed the limit.
    WorkBudgetExceeded {
        /// Which admission stage exceeded its budget.
        phase: LocalInterleavingWorkPhase,
        /// Required work known when admission stopped, saturated on overflow.
        required_at_least: usize,
    },
    /// Checkpoints and one reconstructed block exceed the score-memory limit.
    MemoryBudgetExceeded,
    /// The population of possible addressed matches exceeds its limit.
    CandidateBudgetExceeded,
}

impl LocalInterleavingBudget {
    /// The work-cell limit [`Self::WorkBudgetExceeded`] enforces: witness
    /// scalar cells/packed-word passes plus banded source/reference cells.
    pub const WORK_CELLS: usize = WORK_CELLS;
    /// The retained score-cell limit [`Self::MemoryBudgetExceeded`] enforces.
    pub const SCORE_CELLS: usize = SCORE_CELLS;
    /// The distinct addressed candidate-match limit
    /// [`Self::CandidateBudgetExceeded`] enforces.
    pub const CANDIDATES: usize = CANDIDATES;
}

/// An internal inconsistency of the packed search: what its forward pass or
/// reconstruction found contradicts what admission's feasible witness proved.
/// Each is checked in every build, never only by a debug assertion.
///
/// Serialized as UTR evidence by derive: `{"kind": "optimum_below_feasible",
/// "optimum": .., "feasible": ..}`, one object per variant with its fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LocalInterleavingInconsistency {
    /// The forward pass reached no viable end row covering the whole
    /// reference, although a complete composition exists. Refused rather
    /// than read as an optimum of zero (which would report every word
    /// unmatched).
    NoCompleteComposition,
    /// The end row's score is below the feasible witness's, so the forward
    /// pass lost matches the witness exhibits. Refused rather than reported,
    /// which would call those words missing.
    OptimumBelowFeasible {
        /// The end row's score.
        optimum: usize,
        /// The feasible witness's matched count, a lower bound.
        feasible: usize,
    },
    /// Reconstruction selected a different number of matches than the
    /// optimum it reconstructs.
    ReconstructionMismatch {
        /// Matches reconstruction selected.
        selected: usize,
        /// The optimum it was reconstructing.
        optimum: usize,
    },
    /// The scalar banded traceback reached a step its forward rows do not
    /// explain: the cell is unreachable, or no incoming edge yields its score.
    TracebackStalled {
        /// The reference prefix length where it stalled.
        reference: usize,
    },
    /// A packed row advanced past a node whose required score is more than
    /// one above its base: a word can add at most one match.
    ProfileBaseSkipped {
        /// The row's base score.
        base: usize,
        /// The node's required score.
        required: usize,
    },
    /// Two packed rows merged into one node had different base scores,
    /// although every row reaching a node is trimmed to its required score.
    ProfileBasesDiffer {
        /// The first row's base.
        left: usize,
        /// The second row's base.
        right: usize,
    },
    /// An edge observation's matching-only cut was not immediately before
    /// the suffix domain it extends.
    EdgeCutGap {
        /// The matching-only cut's reference position.
        earlier: usize,
        /// The first position of the suffix domain.
        first: usize,
    },
    /// An edge observation read a forward score outside the forward row's
    /// domain.
    ProfileScoreOutsideDomain {
        /// The reference position read.
        reference: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Address {
    utterance: usize,
    word: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Selected {
    source: usize,
    reference: usize,
}

/// Exact source-bound matching population across legal adjacent-speaker episodes.
pub struct LocalInterleaving<'source> {
    turns: Vec<SpeakerTurn<'source>>,
    reference: &'source [String],
    addresses: Vec<Address>,
    words: Vec<WordEvidence>,
    selected: Vec<Selected>,
    search_frontiers: Vec<SearchFrontier>,
    required: Vec<usize>,
}

/// Read-only addressed correspondence; selection alone is not proof authority.
#[derive(Debug, Clone, Copy)]
pub struct LocalMatch<'source> {
    address: Address,
    reference_index: usize,
    source: &'source str,
    reference: &'source str,
}

impl<'source> LocalMatch<'source> {
    /// Original turn ordinal, never an episode-local index.
    pub fn utterance_index(self) -> usize {
        self.address.utterance
    }
    /// Original within-turn word ordinal.
    pub fn word_index(self) -> usize {
        self.address.word
    }
    /// Original reference word ordinal.
    pub fn reference_index(self) -> usize {
        self.reference_index
    }
    /// Producer-bound source spelling.
    pub fn source_text(self) -> &'source str {
        self.source
    }
    /// Producer-bound reference spelling.
    pub fn reference_text(self) -> &'source str {
        self.reference
    }
}

/// Common to every optimum of the complete source-bound episode composition.
/// Only the alignment producer can construct this state.
///
/// ```compile_fail
/// use batchalign_transform::dp_align::interleaving::{CommonLocalMatch, LocalMatch};
/// fn promote(selected: LocalMatch<'_>) -> CommonLocalMatch<'_> {
///     CommonLocalMatch(selected)
/// }
/// ```
#[derive(Debug, Clone, Copy)]
pub struct CommonLocalMatch<'source>(LocalMatch<'source>);

impl<'source> CommonLocalMatch<'source> {
    /// Inspect correspondence without allowing a selected match to create proof.
    pub fn matched(self) -> LocalMatch<'source> {
        self.0
    }
}

/// Possible, missing and common evidence for one source word.
pub struct LocalWord<'analysis, 'source> {
    analysis: &'analysis LocalInterleaving<'source>,
    source: usize,
}

/// A source word matched in every optimum, possibly at several reference
/// locations. It is not a common correspondence or timing anchor.
pub struct RequiredLocalWord<'analysis, 'source>(LocalWord<'analysis, 'source>);

impl<'source> RequiredLocalWord<'_, 'source> {
    /// Complete nonempty candidate population; no selected-path shortcut.
    pub fn candidates(&self) -> impl Iterator<Item = LocalMatch<'source>> + '_ {
        self.0.candidates()
    }
}

/// Source-order boundaries outside every legal overlap episode containing a
/// turn. Borrowed from the joint producer; a selected path cannot create one.
/// This admits search bounds, not correspondence for the turn's missing words.
pub struct LocalSearchCorridor<'analysis, 'source> {
    analysis: &'analysis LocalInterleaving<'source>,
    before: usize,
    after: usize,
    utterance: usize,
    source_begin: usize,
    source_end: usize,
}

struct SearchFrontier {
    before: usize,
    after: usize,
}

impl<'source> LocalSearchCorridor<'_, 'source> {
    /// Nearest preceding common matches first, all ordered before the turn.
    pub fn preceding(&self) -> impl Iterator<Item = CommonLocalMatch<'source>> + '_ {
        (0..self.before).rev().filter_map(|source| {
            LocalWord {
                analysis: self.analysis,
                source,
            }
            .common()
        })
    }

    /// Nearest following common matches first, all ordered after the turn.
    pub fn following(&self) -> impl Iterator<Item = CommonLocalMatch<'source>> + '_ {
        (self.after..self.analysis.words.len()).filter_map(|source| {
            LocalWord {
                analysis: self.analysis,
                source,
            }
            .common()
        })
    }

    /// Obligatory preceding words, nearest first. Candidate extrema may bound
    /// a search even when no individual correspondence is common.
    pub fn preceding_required(&self) -> impl Iterator<Item = RequiredLocalWord<'_, 'source>> + '_ {
        let end = self
            .analysis
            .required
            .partition_point(|&source| source < self.source_begin);
        self.analysis.required[..end]
            .iter()
            .rev()
            .copied()
            .filter(|&source| source < self.before || self.same_speaker(source))
            .map(|source| {
                RequiredLocalWord(LocalWord {
                    analysis: self.analysis,
                    source,
                })
            })
    }

    /// Obligatory following words, nearest first, with all candidate locations.
    pub fn following_required(&self) -> impl Iterator<Item = RequiredLocalWord<'_, 'source>> + '_ {
        let start = self
            .analysis
            .required
            .partition_point(|&source| source < self.source_end);
        self.analysis.required[start..]
            .iter()
            .copied()
            .filter(|&source| source >= self.after || self.same_speaker(source))
            .map(|source| {
                RequiredLocalWord(LocalWord {
                    analysis: self.analysis,
                    source,
                })
            })
    }

    fn same_speaker(&self, source: usize) -> bool {
        self.analysis.turns[self.analysis.addresses[source].utterance].speaker
            == self.analysis.turns[self.utterance].speaker
    }
}

impl<'analysis, 'source> LocalWord<'analysis, 'source> {
    /// Admit only a nonempty match population present in every optimum.
    pub fn required(self) -> Option<RequiredLocalWord<'analysis, 'source>> {
        let evidence = &self.analysis.words[self.source];
        (!evidence.can_be_missing && !evidence.candidates.is_empty())
            .then_some(RequiredLocalWord(self))
    }
    /// An optimal assignment can omit this word.
    pub fn can_be_missing(&self) -> bool {
        self.analysis.words[self.source].can_be_missing
    }
    /// Every reference location used by some optimal assignment.
    pub fn candidates(&self) -> impl Iterator<Item = LocalMatch<'source>> + '_ {
        self.analysis.words[self.source]
            .candidates
            .iter()
            .map(|&reference| {
                self.analysis.bound(Selected {
                    source: self.source,
                    reference,
                })
            })
    }
    /// A reference match common to all optimal assignments, if proved.
    pub fn common(&self) -> Option<CommonLocalMatch<'source>> {
        let evidence = &self.analysis.words[self.source];
        (!evidence.can_be_missing && evidence.candidates.len() == 1).then(|| {
            CommonLocalMatch(self.analysis.bound(Selected {
                source: self.source,
                reference: evidence.candidates[0],
            }))
        })
    }
}

impl<'source> LocalInterleaving<'source> {
    /// Analyze all legal singleton/adjacent-speaker-run compositions jointly.
    /// Budgets are admitted before allocating the graph or score storage.
    pub fn observe(
        turns: &[SpeakerTurn<'source>],
        reference: &'source [String],
        mode: MatchMode,
    ) -> Result<Self, LocalInterleavingRefusal> {
        uninterrupted(Self::observe_controlled(
            turns,
            reference,
            mode,
            &Uninterrupted,
        ))
    }

    /// The outer result reports interruption; budget refusals remain separate.
    pub fn observe_controlled<C: WorkControl + ?Sized>(
        turns: &[SpeakerTurn<'source>],
        reference: &'source [String],
        mode: MatchMode,
        control: &C,
    ) -> Result<Result<Self, LocalInterleavingRefusal>, C::Error> {
        match projection::SearchProjection::admit(turns, reference, mode, control)? {
            Ok(projection) => projection.observe(control),
            Err(refusal) => Ok(Err(refusal)),
        }
    }

    /// Maximum matched population under the declared bounded order model.
    pub fn matched_words(&self) -> usize {
        self.selected.len()
    }
    /// One globally compatible optimum for inspection, never timing authority.
    pub fn selected(&self) -> impl Iterator<Item = LocalMatch<'source>> + '_ {
        self.selected.iter().map(|&matched| self.bound(matched))
    }
    /// All source words in original document/word order.
    pub fn words(&self) -> impl Iterator<Item = LocalWord<'_, 'source>> {
        (0..self.words.len()).map(|source| LocalWord {
            analysis: self,
            source,
        })
    }
    /// Conservative order corridor, independent of which optimum was selected.
    pub fn search_corridor(&self, utterance: usize) -> Option<LocalSearchCorridor<'_, 'source>> {
        let frontier = self.search_frontiers.get(utterance)?;
        Some(LocalSearchCorridor {
            analysis: self,
            before: frontier.before,
            after: frontier.after,
            utterance,
            source_begin: self.addresses.partition_point(|a| a.utterance < utterance),
            source_end: self.addresses.partition_point(|a| a.utterance <= utterance),
        })
    }
    fn bound(&self, matched: Selected) -> LocalMatch<'source> {
        let address = self.addresses[matched.source];
        LocalMatch {
            address,
            reference_index: matched.reference,
            source: &self.turns[address.utterance].words[address.word],
            reference: &self.reference[matched.reference],
        }
    }
}

/// An episode preserves the first turn and the following speaker's turn/word
/// order as two chains. Every prefix of that following run is a legal choice;
/// choosing one consumes the whole episode before the next source boundary.
struct Episode {
    first: usize,
    following_end: usize,
}

fn episodes<'a>(turns: &'a [SpeakerTurn<'_>]) -> impl Iterator<Item = Episode> + 'a {
    (0..turns.len().saturating_sub(1)).flat_map(|first| {
        let mut end = first + 1;
        if turns[first].speaker != turns[end].speaker {
            let speaker = turns[end].speaker;
            end += 1;
            while end < turns.len()
                && turns[end].speaker == speaker
                && turns[end].continuation == Continuation::OverlappingSpeakerRun
            {
                end += 1;
            }
        }
        (first + 2..=end).map(move |following_end| Episode {
            first,
            following_end,
        })
    })
}

enum Relation<'source> {
    Literal {
        source: Vec<&'source str>,
        reference: &'source [String],
        mode: LiteralMatchMode,
    },
    Fuzzy {
        source: Vec<PreparedFuzzyWord<'source>>,
        reference: Vec<PreparedFuzzyWord<'source>>,
        policy: FuzzyComparison,
    },
}

impl<'source> Relation<'source> {
    fn new(source: Vec<&'source str>, reference: &'source [String], mode: MatchMode) -> Self {
        match mode {
            MatchMode::Fuzzy { threshold } => Self::Fuzzy {
                source: source.into_iter().map(PreparedFuzzyWord::new).collect(),
                reference: reference
                    .iter()
                    .map(|word| PreparedFuzzyWord::new(word))
                    .collect(),
                policy: FuzzyComparison::new(threshold),
            },
            MatchMode::Exact => Self::Literal {
                source,
                reference,
                mode: LiteralMatchMode::Exact,
            },
            MatchMode::CaseInsensitive => Self::Literal {
                source,
                reference,
                mode: LiteralMatchMode::AsciiInsensitive,
            },
        }
    }
    fn matches(&self, word: usize, k: usize) -> bool {
        match self {
            Self::Literal {
                source,
                reference,
                mode,
            } => mode.matches_words(source[word], &reference[k]),
            Self::Fuzzy {
                source,
                reference,
                policy,
            } => source[word].matches(&reference[k], *policy),
        }
    }
}

#[cfg(test)]
mod tests;
