//! Joint bounded adjacent-turn composition. Every complete source-DAG path
//! consumes every source word once, through singleton turns or disjoint episodes
//! of one turn and a following run of another speaker. Reference words are consumed once.
//! Independent local proposals cannot reserve the same ASR word twice here.
//!
//! Reference-layer checkpoints permit exact forward/backward proof without
//! keeping a recording-wide source-state × reference matrix in memory.

use super::{Alignable, FuzzyComparison, MatchMode, PreparedFuzzyWord, WordEvidence};

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

/// Bounded analysis refusal, never an assertion of absence or ambiguity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocalInterleavingRefusal {
    /// Checked source-DAG × reference work exceeds the limit.
    WorkBudgetExceeded,
    /// Checkpoints and one reconstructed block exceed the score-memory limit.
    MemoryBudgetExceeded,
    /// The population of possible addressed matches exceeds its limit.
    CandidateBudgetExceeded,
}

impl LocalInterleavingRefusal {
    /// The production limit the refused budget enforces, in its own unit:
    /// source-DAG nodes times reference positions, retained score cells, or
    /// distinct addressed candidate matches.
    pub fn limit(self) -> usize {
        match self {
            Self::WorkBudgetExceeded => WORK_CELLS,
            Self::MemoryBudgetExceeded => SCORE_CELLS,
            Self::CandidateBudgetExceeded => CANDIDATES,
        }
    }
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
    optimum: usize,
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
        let budget = Budget::admit(turns, reference.len())?;
        let (graph, addresses) = Graph::of(turns, budget.nodes);
        let source: Vec<_> = addresses
            .iter()
            .map(|a| turns[a.utterance].words[a.word].as_str())
            .collect();
        let relation = Relation::new(source, reference, mode);
        let mut row = vec![0usize; graph.edges.len()];
        let mut checkpoints = vec![row.clone()];
        for k in 0..reference.len() {
            row = graph.forward(&row, k, &relation);
            if (k + 1) % budget.block == 0 {
                checkpoints.push(row.clone());
            }
        }
        let optimum = row[graph.end];
        let mut words = addresses
            .iter()
            .map(|_| WordEvidence {
                candidates: Vec::new(),
                can_be_missing: false,
            })
            .collect::<Vec<_>>();
        let mut backward = vec![0usize; graph.edges.len()];
        graph.observe_edges(&row, &backward, None, &relation, optimum, &mut words);
        let mut candidate_count = 0usize;
        let mut end = reference.len();
        // Backward scores prove all optimal correspondences. Traceback uses
        // the same source graph and retained forward checkpoints, never an
        // independent monotonic rematch.
        while end > 0 {
            let start = (end - 1) / budget.block * budget.block;
            let mut block = Vec::with_capacity(end - start + 1);
            block.push(checkpoints[start / budget.block].clone());
            for k in start..end {
                block.push(graph.forward(&block[k - start], k, &relation));
            }
            for k in (start..end).rev() {
                let next = backward;
                backward = graph.backward(&next, k, &relation);
                graph.observe_edges(
                    &block[k - start],
                    &backward,
                    Some((&next, k)),
                    &relation,
                    optimum,
                    &mut words,
                );
                // A source edge can occur in multiple episode states. Count
                // distinct source/reference pairs, not duplicate graph edges.
                for word in &mut words {
                    if word.candidates.last() == Some(&k) {
                        candidate_count += 1;
                    }
                }
                if candidate_count > CANDIDATES {
                    return Err(LocalInterleavingRefusal::CandidateBudgetExceeded);
                }
            }
            end = start;
        }
        for word in &mut words {
            word.candidates.reverse();
        }
        // Common proof does not depend on traceback. Select one optimum using
        // the retained forward checkpoints, recomputing only a bounded block.
        let selected = graph.traceback(&checkpoints, budget.block, reference.len(), &relation);
        let required = words
            .iter()
            .enumerate()
            .filter_map(|(index, evidence)| {
                (!evidence.can_be_missing && !evidence.candidates.is_empty()).then_some(index)
            })
            .collect();
        let search_frontiers = search_frontiers(turns)
            .into_iter()
            .map(|frontier| SearchFrontier {
                before: addresses.partition_point(|a| a.utterance < frontier.before),
                after: addresses.partition_point(|a| a.utterance < frontier.after),
            })
            .collect();
        Ok(Self {
            turns: turns.to_vec(),
            reference,
            addresses,
            words,
            selected,
            optimum,
            search_frontiers,
            required,
        })
    }

    /// Maximum matched population under the declared bounded order model.
    pub fn matched_words(&self) -> usize {
        self.optimum
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

struct Budget {
    nodes: usize,
    block: usize,
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

// Use the same episode owner as Graph::of. A sweep retains the outermost
// episodes containing each turn without expanding every range (O(turns+episodes)).
fn search_frontiers(turns: &[SpeakerTurn<'_>]) -> Vec<SearchFrontier> {
    let mut ends: Vec<_> = (0..turns.len()).map(|i| i + 1).collect();
    for episode in episodes(turns) {
        ends[episode.first] = ends[episode.first].max(episode.following_end);
    }
    let mut left = 0;
    let mut right = 0;
    ends.iter()
        .enumerate()
        .map(|(ordinal, &end)| {
            // ends[ordinal] >= ordinal+1, so this cursor cannot pass the current
            // turn. Expired intervals need no queue or sentinel boundary.
            while ends[left] <= ordinal {
                left += 1;
            }
            right = right.max(end);
            SearchFrontier {
                before: left,
                after: right,
            }
        })
        .collect()
}

impl Budget {
    fn admit(
        turns: &[SpeakerTurn<'_>],
        reference: usize,
    ) -> Result<Self, LocalInterleavingRefusal> {
        let refused = LocalInterleavingRefusal::WorkBudgetExceeded;
        let mut nodes = turns.len().checked_add(1).ok_or(refused)?;
        for turn in turns {
            nodes = nodes
                .checked_add(turn.words.len().saturating_sub(1))
                .ok_or(refused)?;
        }
        for episode in episodes(turns) {
            let second = turns[episode.first + 1..episode.following_end]
                .iter()
                .try_fold(0usize, |count, turn| count.checked_add(turn.words.len()))
                .ok_or(refused)?;
            let states = turns[episode.first]
                .words
                .len()
                .checked_add(1)
                .and_then(|a| second.checked_add(1).and_then(|b| a.checked_mul(b)))
                .ok_or(refused)?;
            nodes = nodes.checked_add(states.saturating_sub(2)).ok_or(refused)?;
        }
        if nodes
            .checked_mul(reference.checked_add(1).ok_or(refused)?)
            .is_none_or(|work| work > WORK_CELLS)
        {
            return Err(refused);
        }
        let mut block = 1usize;
        while block < reference.div_ceil(block) {
            block += 1;
        }
        let layers = reference / block + block + 6;
        if nodes
            .checked_mul(layers)
            .is_none_or(|cells| cells > SCORE_CELLS)
        {
            return Err(LocalInterleavingRefusal::MemoryBudgetExceeded);
        }
        Ok(Self { nodes, block })
    }
}

#[derive(Clone, Copy)]
struct Edge {
    to: usize,
    word: Option<usize>,
}

struct Graph {
    edges: Vec<Vec<Edge>>,
    incoming: Vec<Vec<Edge>>,
    order: Vec<usize>,
    end: usize,
}

impl Graph {
    fn of(turns: &[SpeakerTurn<'_>], capacity: usize) -> (Self, Vec<Address>) {
        let mut edges = vec![Vec::new(); turns.len() + 1];
        edges.reserve(capacity - edges.len());
        let mut addresses = Vec::new();
        let mut offsets = Vec::with_capacity(turns.len());
        for (utterance, turn) in turns.iter().enumerate() {
            offsets.push(addresses.len());
            addresses.extend((0..turn.words.len()).map(|word| Address { utterance, word }));
            let mut node = utterance;
            for word in 0..turn.words.len() {
                let next = if word + 1 == turn.words.len() {
                    utterance + 1
                } else {
                    edges.push(Vec::new());
                    edges.len() - 1
                };
                edges[node].push(Edge {
                    to: next,
                    word: Some(offsets[utterance] + word),
                });
                node = next;
            }
            if turn.words.is_empty() {
                edges[node].push(Edge {
                    to: utterance + 1,
                    word: None,
                });
            }
        }
        for episode in episodes(turns) {
            let utterance = episode.first;
            let second_start = offsets[utterance + 1];
            let second_end = offsets
                .get(episode.following_end)
                .copied()
                .unwrap_or(addresses.len());
            let (a, b) = (turns[utterance].words.len(), second_end - second_start);
            if a == 0 && b == 0 {
                edges[utterance].push(Edge {
                    to: episode.following_end,
                    word: None,
                });
                continue;
            }
            let mut grid = Vec::with_capacity((a + 1) * (b + 1));
            for i in 0..=a {
                for j in 0..=b {
                    grid.push(if i == 0 && j == 0 {
                        utterance
                    } else if i == a && j == b {
                        episode.following_end
                    } else {
                        edges.push(Vec::new());
                        edges.len() - 1
                    });
                }
            }
            for i in 0..=a {
                for j in 0..=b {
                    let node = grid[i * (b + 1) + j];
                    if i < a {
                        edges[node].push(Edge {
                            to: grid[(i + 1) * (b + 1) + j],
                            word: Some(offsets[utterance] + i),
                        });
                    }
                    if j < b {
                        edges[node].push(Edge {
                            to: grid[i * (b + 1) + j + 1],
                            word: Some(second_start + j),
                        });
                    }
                }
            }
        }
        let mut incoming = vec![Vec::new(); edges.len()];
        for (from, outgoing) in edges.iter().enumerate() {
            for edge in outgoing {
                incoming[edge.to].push(Edge {
                    to: from,
                    word: edge.word,
                });
            }
        }
        let mut degree: Vec<_> = incoming.iter().map(Vec::len).collect();
        let mut order = vec![0];
        let mut cursor = 0;
        while cursor < order.len() {
            for edge in &edges[order[cursor]] {
                degree[edge.to] -= 1;
                if degree[edge.to] == 0 {
                    order.push(edge.to);
                }
            }
            cursor += 1;
        }
        (
            Self {
                edges,
                incoming,
                order,
                end: turns.len(),
            },
            addresses,
        )
    }

    fn forward(&self, previous: &[usize], k: usize, relation: &Relation<'_>) -> Vec<usize> {
        let mut row = previous.to_vec();
        for &from in &self.order {
            for edge in &self.edges[from] {
                let matched = edge.word.is_some_and(|word| relation.matches(word, k));
                row[edge.to] = row[edge.to].max(row[from]);
                if matched {
                    row[edge.to] = row[edge.to].max(previous[from] + 1);
                }
            }
        }
        row
    }

    fn backward(&self, next: &[usize], k: usize, relation: &Relation<'_>) -> Vec<usize> {
        let mut row = next.to_vec();
        for &from in self.order.iter().rev() {
            for edge in &self.edges[from] {
                row[from] = row[from].max(row[edge.to]);
                if edge.word.is_some_and(|word| relation.matches(word, k)) {
                    row[from] = row[from].max(next[edge.to] + 1);
                }
            }
        }
        row
    }

    fn observe_edges(
        &self,
        forward: &[usize],
        backward: &[usize],
        next: Option<(&[usize], usize)>,
        relation: &Relation<'_>,
        optimum: usize,
        words: &mut [WordEvidence],
    ) {
        for &from in &self.order {
            for edge in &self.edges[from] {
                let Some(word) = edge.word else {
                    continue;
                };
                if forward[from] + backward[edge.to] == optimum {
                    words[word].can_be_missing = true;
                }
                if let Some((next, k)) = next
                    && relation.matches(word, k)
                    && forward[from] + 1 + next[edge.to] == optimum
                    && words[word].candidates.last() != Some(&k)
                {
                    words[word].candidates.push(k);
                }
            }
        }
    }

    fn traceback(
        &self,
        checkpoints: &[Vec<usize>],
        width: usize,
        reference_len: usize,
        relation: &Relation<'_>,
    ) -> Vec<Selected> {
        let mut node = self.end;
        let mut k = reference_len;
        let mut selected = Vec::new();
        while k > 0 {
            let start = (k - 1) / width * width;
            let mut rows = vec![checkpoints[start / width].clone()];
            for reference in start..k {
                rows.push(self.forward(&rows[reference - start], reference, relation));
            }
            while k > start {
                let current = &rows[k - start];
                let previous = &rows[k - start - 1];
                if current[node] == previous[node] {
                    k -= 1;
                    continue;
                }
                let mut advanced = false;
                for edge in &self.incoming[node] {
                    if let Some(word) = edge.word
                        && relation.matches(word, k - 1)
                        && previous[edge.to] + 1 == current[node]
                    {
                        selected.push(Selected {
                            source: word,
                            reference: k - 1,
                        });
                        node = edge.to;
                        k -= 1;
                        advanced = true;
                        break;
                    }
                    if current[edge.to] == current[node] {
                        node = edge.to;
                        advanced = true;
                        break;
                    }
                }
                // Source graph construction guarantees an optimal incoming edge
                // whenever the reference skip is not optimal. No external graph
                // or score table can enter this producer.
                debug_assert!(advanced);
            }
        }
        selected.reverse();
        selected
    }
}

enum Relation<'source> {
    Literal {
        source: Vec<&'source str>,
        reference: &'source [String],
        mode: MatchMode,
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
            _ => Self::Literal {
                source,
                reference,
                mode,
            },
        }
    }
    fn matches(&self, word: usize, k: usize) -> bool {
        match self {
            Self::Literal {
                source,
                reference,
                mode,
            } => match mode {
                MatchMode::Exact => source[word] == reference[k],
                _ => source[word].eq_ignore_ascii_case(&reference[k]),
            },
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
