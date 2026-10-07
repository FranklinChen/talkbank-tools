//! Bounded lexical correspondence for two independently ordered speakers.
//!
//! A reference word can be consumed only once. Each speaker's source order is
//! retained, but no order is imposed between speakers. This is not two separate
//! alignments: independent alignments could both claim the same reference word.
//! Forward and backward scores on the product DAG describe *all* maximum-match
//! assignments. A selected assignment is inspection evidence, not certainty.
//!
//! This kernel admits lexical correspondence, never acoustic speaker identity,
//! a timing interval or permission to write CHAT. Results retain their borrowed
//! inputs so addresses cannot silently be paired with a different word census.

use super::MatchMode;
use super::comparison::{Alignable, FuzzyComparison, PreparedFuzzyWord};

mod episodes;
pub use episodes::{
    CommonLocalMatch, LocalInterleaving, LocalInterleavingRefusal, LocalMatch, LocalSearchCorridor,
    LocalWord, RequiredLocalWord, SpeakerTurn,
};

const MAX_CELLS: usize = 1_048_576;

/// Which independently ordered source a word belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SpeakerChain {
    /// First speaker's ordered words.
    First,
    /// Second speaker's ordered words.
    Second,
}

impl SpeakerChain {
    fn index(self) -> usize {
        match self {
            Self::First => 0,
            Self::Second => 1,
        }
    }
}

/// Proof work could not be performed; this says nothing about ambiguity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterleavingRefusal {
    /// Product-DAG size overflowed or exceeded the fixed work/memory budget.
    CellBudgetExceeded,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct MatchAddress {
    chain: SpeakerChain,
    word: usize,
    reference: usize,
}

struct WordEvidence {
    candidates: Vec<usize>,
    can_be_missing: bool,
}

/// Complete bounded analysis, inseparable from the source and reference words.
pub struct TwoSpeakerAlignment<'source> {
    sources: [&'source [String]; 2],
    reference: &'source [String],
    optimum: usize,
    selected: Vec<MatchAddress>,
    words: [Vec<WordEvidence>; 2],
}

/// A producer-bound match, carrying the exact words its addresses describe.
/// Consumers cannot construct one from a selected index or a certainty flag.
#[derive(Debug, Clone, Copy)]
pub struct InterleavingMatch<'source> {
    address: MatchAddress,
    source: &'source str,
    reference: &'source str,
}

/// Lexical correspondence common to every optimum of this bound two-chain
/// analysis. A selected or possible match cannot construct this proof state.
///
/// ```compile_fail
/// use batchalign_transform::dp_align::{MatchMode, interleaving::{
///     CommonInterleavingMatch, TwoSpeakerAlignment,
/// }};
/// let source = vec!["yes".to_owned()];
/// let observation = TwoSpeakerAlignment::observe(&source, &source, &source, MatchMode::Exact).unwrap();
/// let selected = observation.selected().next().unwrap();
/// let _: CommonInterleavingMatch<'_> = selected;
/// ```
#[derive(Debug, Clone, Copy)]
pub struct CommonInterleavingMatch<'source>(InterleavingMatch<'source>);

impl<'source> CommonInterleavingMatch<'source> {
    /// Inspect the addressed words without giving selected paths proof authority.
    pub fn matched(self) -> InterleavingMatch<'source> {
        self.0
    }
}

impl<'source> InterleavingMatch<'source> {
    /// Source speaker chain.
    pub fn chain(self) -> SpeakerChain {
        self.address.chain
    }
    /// Word's ordinal within that speaker chain.
    pub fn word_index(self) -> usize {
        self.address.word
    }
    /// Word's ordinal in this analysis's reference.
    pub fn reference_index(self) -> usize {
        self.address.reference
    }
    /// Source spelling retained by the producer.
    pub fn source_text(self) -> &'source str {
        self.source
    }
    /// Reference spelling retained by the producer.
    pub fn reference_text(self) -> &'source str {
        self.reference
    }
}

/// Read-only evidence for one source word across every optimal assignment.
pub struct InterleavingWord<'analysis, 'source> {
    analysis: &'analysis TwoSpeakerAlignment<'source>,
    chain: SpeakerChain,
    word: usize,
}

impl<'source> InterleavingWord<'_, 'source> {
    fn evidence(&self) -> &WordEvidence {
        &self.analysis.words[self.chain.index()][self.word]
    }

    /// At least one optimal assignment omits this source word.
    pub fn can_be_missing(&self) -> bool {
        self.evidence().can_be_missing
    }

    /// All possible reference matches, not an admitted timing or search window.
    pub fn candidates(&self) -> impl Iterator<Item = InterleavingMatch<'source>> + '_ {
        self.evidence().candidates.iter().map(|&reference| {
            self.analysis.bound(MatchAddress {
                chain: self.chain,
                word: self.word,
                reference,
            })
        })
    }

    /// Correspondence common to every optimal assignment, if there is one.
    pub fn common(&self) -> Option<CommonInterleavingMatch<'source>> {
        let evidence = self.evidence();
        if evidence.can_be_missing || evidence.candidates.len() != 1 {
            return None;
        }
        Some(CommonInterleavingMatch(self.analysis.bound(MatchAddress {
            chain: self.chain,
            word: self.word,
            reference: evidence.candidates[0],
        })))
    }
}

impl<'source> TwoSpeakerAlignment<'source> {
    /// Analyze both chains jointly with the existing alignment match relation.
    /// Size admission precedes allocation and fuzzy-comparison preparation.
    pub fn observe(
        first: &'source [String],
        second: &'source [String],
        reference: &'source [String],
        mode: MatchMode,
    ) -> Result<Self, InterleavingRefusal> {
        Self::with_budget(first, second, reference, mode, MAX_CELLS)
    }

    /// Maximum matched lexical population, not an acoustic-quality score.
    pub fn matched_words(&self) -> usize {
        self.optimum
    }

    /// One optimal assignment for inspection only.
    pub fn selected(&self) -> impl Iterator<Item = InterleavingMatch<'source>> + '_ {
        self.selected.iter().map(|&address| self.bound(address))
    }

    /// Per-word possible, missing and common evidence in source order.
    pub fn words(
        &self,
        chain: SpeakerChain,
    ) -> impl Iterator<Item = InterleavingWord<'_, 'source>> {
        (0..self.sources[chain.index()].len()).map(move |word| InterleavingWord {
            analysis: self,
            chain,
            word,
        })
    }

    fn bound(&self, address: MatchAddress) -> InterleavingMatch<'source> {
        InterleavingMatch {
            address,
            source: &self.sources[address.chain.index()][address.word],
            reference: &self.reference[address.reference],
        }
    }

    fn with_budget(
        first: &'source [String],
        second: &'source [String],
        reference: &'source [String],
        mode: MatchMode,
        budget: usize,
    ) -> Result<Self, InterleavingRefusal> {
        let shape = AdmittedShape::new(first.len(), second.len(), reference.len(), budget)
            .ok_or(InterleavingRefusal::CellBudgetExceeded)?;
        let relation = MatchRelation::new([first, second], reference, mode);
        let mut forward = vec![0usize; shape.cells];
        for i in 0..=first.len() {
            for j in 0..=second.len() {
                for k in 0..=reference.len() {
                    let mut score = 0;
                    if i > 0 {
                        score = score.max(forward[shape.at(i - 1, j, k)]);
                    }
                    if j > 0 {
                        score = score.max(forward[shape.at(i, j - 1, k)]);
                    }
                    if k > 0 {
                        score = score.max(forward[shape.at(i, j, k - 1)]);
                        if i > 0 && relation.matches(SpeakerChain::First, i - 1, k - 1) {
                            score = score.max(forward[shape.at(i - 1, j, k - 1)] + 1);
                        }
                        if j > 0 && relation.matches(SpeakerChain::Second, j - 1, k - 1) {
                            score = score.max(forward[shape.at(i, j - 1, k - 1)] + 1);
                        }
                    }
                    forward[shape.at(i, j, k)] = score;
                }
            }
        }
        let mut backward = vec![0usize; shape.cells];
        for i in (0..=first.len()).rev() {
            for j in (0..=second.len()).rev() {
                for k in (0..=reference.len()).rev() {
                    let mut score = 0;
                    if i < first.len() {
                        score = score.max(backward[shape.at(i + 1, j, k)]);
                    }
                    if j < second.len() {
                        score = score.max(backward[shape.at(i, j + 1, k)]);
                    }
                    if k < reference.len() {
                        score = score.max(backward[shape.at(i, j, k + 1)]);
                        if i < first.len() && relation.matches(SpeakerChain::First, i, k) {
                            score = score.max(backward[shape.at(i + 1, j, k + 1)] + 1);
                        }
                        if j < second.len() && relation.matches(SpeakerChain::Second, j, k) {
                            score = score.max(backward[shape.at(i, j + 1, k + 1)] + 1);
                        }
                    }
                    backward[shape.at(i, j, k)] = score;
                }
            }
        }
        let optimum = backward[0];
        let mut words = [first, second].map(|source| {
            (0..source.len())
                .map(|_| WordEvidence {
                    candidates: Vec::new(),
                    can_be_missing: false,
                })
                .collect::<Vec<_>>()
        });
        // Inspect all edges on an optimal path, not just the chosen traceback.
        // Repeated edges caused by another chain's state are deduplicated below.
        for i in 0..=first.len() {
            for j in 0..=second.len() {
                for k in 0..=reference.len() {
                    let prefix = forward[shape.at(i, j, k)];
                    for (chain, word, next_i, next_j) in [
                        (SpeakerChain::First, i, i + 1, j),
                        (SpeakerChain::Second, j, i, j + 1),
                    ] {
                        if word == words[chain.index()].len() {
                            continue;
                        }
                        let evidence = &mut words[chain.index()][word];
                        if prefix + backward[shape.at(next_i, next_j, k)] == optimum {
                            evidence.can_be_missing = true;
                        }
                        if k < reference.len()
                            && relation.matches(chain, word, k)
                            && prefix + 1 + backward[shape.at(next_i, next_j, k + 1)] == optimum
                        {
                            evidence.candidates.push(k);
                        }
                    }
                }
            }
        }
        for chain in &mut words {
            for word in chain {
                word.candidates.sort_unstable();
                word.candidates.dedup();
            }
        }
        let mut selected = Vec::with_capacity(optimum);
        let (mut i, mut j, mut k) = (0, 0, 0);
        while i < first.len() || j < second.len() || k < reference.len() {
            let score = backward[shape.at(i, j, k)];
            if i < first.len()
                && k < reference.len()
                && relation.matches(SpeakerChain::First, i, k)
                && 1 + backward[shape.at(i + 1, j, k + 1)] == score
            {
                selected.push(MatchAddress {
                    chain: SpeakerChain::First,
                    word: i,
                    reference: k,
                });
                i += 1;
                k += 1;
            } else if j < second.len()
                && k < reference.len()
                && relation.matches(SpeakerChain::Second, j, k)
                && 1 + backward[shape.at(i, j + 1, k + 1)] == score
            {
                selected.push(MatchAddress {
                    chain: SpeakerChain::Second,
                    word: j,
                    reference: k,
                });
                j += 1;
                k += 1;
            } else if k < reference.len() && backward[shape.at(i, j, k + 1)] == score {
                k += 1;
            } else if i < first.len() && backward[shape.at(i + 1, j, k)] == score {
                i += 1;
            } else {
                // At least one optimal outgoing edge exists at every nonterminal
                // DAG state. The preceding branches cover all but skip-second.
                j += 1;
            }
        }
        Ok(Self {
            sources: [first, second],
            reference,
            optimum,
            selected,
            words,
        })
    }
}

/// Checked dimensions bound both score tables and relation preparation.
struct AdmittedShape {
    second: usize,
    reference: usize,
    cells: usize,
}

impl AdmittedShape {
    fn new(first: usize, second: usize, reference: usize, budget: usize) -> Option<Self> {
        let second = second.checked_add(1)?;
        let reference = reference.checked_add(1)?;
        let cells = first
            .checked_add(1)?
            .checked_mul(second)?
            .checked_mul(reference)?;
        (cells <= budget).then_some(Self {
            second,
            reference,
            cells,
        })
    }

    fn at(&self, first: usize, second: usize, reference: usize) -> usize {
        (first * self.second + second) * self.reference + reference
    }
}

/// Prepare the existing relation once, never lowercase inside DAG traversal.
struct MatchRelation {
    reference_len: usize,
    matches: [Vec<bool>; 2],
}

impl MatchRelation {
    fn new(sources: [&[String]; 2], reference: &[String], mode: MatchMode) -> Self {
        let matches = match mode {
            MatchMode::Exact | MatchMode::CaseInsensitive => sources.map(|source| {
                source
                    .iter()
                    .flat_map(|word| {
                        reference.iter().map(move |other| match mode {
                            MatchMode::Exact => word == other,
                            _ => word.eq_ignore_ascii_case(other),
                        })
                    })
                    .collect()
            }),
            MatchMode::Fuzzy { threshold } => {
                let reference: Vec<_> = reference
                    .iter()
                    .map(|word| PreparedFuzzyWord::new(word))
                    .collect();
                let policy = FuzzyComparison::new(threshold);
                sources.map(|source| {
                    source
                        .iter()
                        .flat_map(|word| {
                            let word = PreparedFuzzyWord::new(word);
                            reference
                                .iter()
                                .map(move |other| word.matches(other, policy))
                        })
                        .collect()
                })
            }
        };
        Self {
            reference_len: reference.len(),
            matches,
        }
    }

    fn matches(&self, chain: SpeakerChain, word: usize, reference: usize) -> bool {
        self.matches[chain.index()][word * self.reference_len + reference]
    }
}

#[cfg(test)]
mod tests;
