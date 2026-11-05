//! Input-owned graph preparation; geometry cannot be supplied independently.

use super::*;
mod observation;

pub(super) struct PreparedGraph<'plan, 'source> {
    turns: &'plan [SpeakerTurn<'source>],
    nodes: usize,
    /// One real source-word boundary before/after each turn, including the end.
    offsets: Vec<usize>,
    episodes: Vec<PreparedEpisode>,
}

enum PreparedEpisode {
    Empty(Episode),
    Grid {
        span: Episode,
        left: usize,
        right: usize,
        cells: usize,
    },
}

impl PreparedEpisode {
    fn span(&self) -> &Episode {
        match self {
            Self::Empty(span) | Self::Grid { span, .. } => span,
        }
    }
}

pub(super) struct MaterializedGraph<'plan, 'source> {
    turns: &'plan [SpeakerTurn<'source>],
    graph: Graph,
    addresses: Vec<Address>,
    frontiers: PendingTurnFrontiers,
}

pub(super) struct CompletedGraphObservation {
    data: GraphObservationData,
}

pub(super) struct GraphObservationData {
    pub(super) words: Vec<WordEvidence>,
    pub(super) selected: Vec<Selected>,
    pub(super) frontiers: Vec<TurnFrontier>,
}

impl CompletedGraphObservation {
    pub(super) fn into_data(self) -> GraphObservationData {
        self.data
    }
}

#[cfg(test)]
impl MaterializedGraph<'_, '_> {
    pub(super) fn graph(&self) -> &Graph {
        &self.graph
    }
    pub(super) fn addresses(&self) -> &[Address] {
        &self.addresses
    }
}

/// A turn-space range, distinct from completed word-space search frontiers.
pub(super) struct TurnFrontier {
    pub(super) before: usize,
    pub(super) after: usize,
}

/// Paid episode classification survives construction; the optional sweep runs
/// only when complete search evidence needs source corridors.
struct PendingTurnFrontiers {
    turns: usize,
    episodes: Vec<PreparedEpisode>,
}

impl PendingTurnFrontiers {
    fn finish<C: WorkControl + ?Sized>(self, control: &C) -> Result<Vec<TurnFrontier>, C::Error> {
        control.check()?;
        let mut ends: Vec<_> = (0..self.turns).map(|i| i + 1).collect();
        for planned in self.episodes {
            control.check()?;
            let episode = planned.span();
            ends[episode.first] = ends[episode.first].max(episode.following_end);
        }
        let mut left = 0;
        let mut right = 0;
        ends.iter()
            .enumerate()
            .map(|(ordinal, &end)| {
                control.check()?;
                while ends[left] <= ordinal {
                    left += 1;
                }
                right = right.max(end);
                Ok(TurnFrontier {
                    before: left,
                    after: right,
                })
            })
            .collect()
    }
}

impl<'plan, 'source> PreparedGraph<'plan, 'source> {
    pub(super) fn admit(
        turns: &'plan [SpeakerTurn<'source>],
    ) -> Result<Self, LocalInterleavingRefusal> {
        let overflow =
            LocalInterleavingRefusal::Budget(LocalInterleavingBudget::WorkBudgetExceeded {
                phase: LocalInterleavingWorkPhase::GraphShape,
                required_at_least: usize::MAX,
            });
        let memory =
            LocalInterleavingRefusal::Budget(LocalInterleavingBudget::MemoryBudgetExceeded);
        let mut nodes = turns.len().checked_add(1).ok_or(overflow)?;
        // Retain the existing legacy policy; this is not a host-memory lease.
        if nodes > SCORE_CELLS / 6 {
            return Err(memory);
        }
        let mut offsets = Vec::with_capacity(turns.len() + 1);
        offsets.push(0);
        let mut words = 0usize;
        for turn in turns {
            words = words.checked_add(turn.words.len()).ok_or(overflow)?;
            offsets.push(words);
            nodes = nodes
                .checked_add(turn.words.len().saturating_sub(1))
                .ok_or(overflow)?;
        }
        if nodes > SCORE_CELLS / 6 {
            return Err(memory);
        }
        let mut prepared = Vec::new();
        for span in episodes(turns) {
            let left = turns[span.first].words.len();
            let right = offsets[span.following_end] - offsets[span.first + 1];
            if left == 0 && right == 0 {
                prepared.push(PreparedEpisode::Empty(span));
            } else {
                let cells = left
                    .checked_add(1)
                    .and_then(|a| right.checked_add(1).and_then(|b| a.checked_mul(b)))
                    .ok_or(overflow)?;
                nodes = nodes.checked_add(cells - 2).ok_or(overflow)?;
                if nodes > SCORE_CELLS / 6 {
                    return Err(memory);
                }
                prepared.push(PreparedEpisode::Grid {
                    span,
                    left,
                    right,
                    cells,
                });
            }
        }
        Ok(Self {
            turns,
            nodes,
            offsets,
            episodes: prepared,
        })
    }

    pub(super) fn materialize<C: WorkControl + ?Sized>(
        self,
        control: &C,
    ) -> Result<MaterializedGraph<'plan, 'source>, C::Error> {
        control.check()?;
        let Self {
            turns,
            nodes,
            offsets,
            episodes,
        } = self;
        let mut edges = Vec::with_capacity(nodes);
        edges.resize_with(turns.len() + 1, Vec::new);
        let mut addresses = Vec::with_capacity(offsets[turns.len()]);
        for (utterance, turn) in turns.iter().enumerate() {
            control.check()?;
            let mut node = utterance;
            for word in 0..turn.words.len() {
                control.check()?;
                addresses.push(Address { utterance, word });
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
        for planned in &episodes {
            control.check()?;
            let episode = planned.span();
            let utterance = episode.first;
            let second_start = offsets[utterance + 1];
            let (a, b, cells) = match planned {
                PreparedEpisode::Empty(_) => {
                    edges[utterance].push(Edge {
                        to: episode.following_end,
                        word: None,
                    });
                    continue;
                }
                PreparedEpisode::Grid {
                    left, right, cells, ..
                } => (*left, *right, *cells),
            };
            let mut grid = Vec::with_capacity(cells);
            for i in 0..=a {
                for j in 0..=b {
                    control.check()?;
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
                    control.check()?;
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
        let mut incoming = Vec::with_capacity(edges.len());
        incoming.resize_with(edges.len(), Vec::new);
        for (from, outgoing) in edges.iter().enumerate() {
            control.check()?;
            for edge in outgoing {
                control.check()?;
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
            control.check()?;
            for edge in &edges[order[cursor]] {
                control.check()?;
                degree[edge.to] -= 1;
                if degree[edge.to] == 0 {
                    order.push(edge.to);
                }
            }
            cursor += 1;
        }
        // Every complete path consumes all source words once. Consequently a
        // node has one consumed-word depth, even when reached via an episode.
        let mut depth = vec![0; edges.len()];
        for &from in &order {
            control.check()?;
            for edge in &edges[from] {
                control.check()?;
                depth[edge.to] = depth[from] + usize::from(edge.word.is_some());
            }
        }
        // Stable sorting retains topological order among epsilon-linked nodes.
        order.sort_by_key(|&node| depth[node]);
        let mut rank = vec![0; edges.len()];
        for (slot, &node) in order.iter().enumerate() {
            control.check()?;
            rank[node] = slot;
        }
        Ok(MaterializedGraph {
            turns,
            graph: Graph {
                edges,
                incoming,
                order,
                rank,
                depth,
                end: turns.len(),
            },
            addresses,
            frontiers: PendingTurnFrontiers {
                turns: turns.len(),
                episodes,
            },
        })
    }
}

pub(super) struct Edge {
    pub(super) to: usize,
    pub(super) word: Option<usize>,
}

pub(super) struct Graph {
    pub(super) edges: Vec<Vec<Edge>>,
    pub(super) incoming: Vec<Vec<Edge>>,
    pub(super) order: Vec<usize>,
    pub(super) rank: Vec<usize>,
    pub(super) depth: Vec<usize>,
    pub(super) end: usize,
}
