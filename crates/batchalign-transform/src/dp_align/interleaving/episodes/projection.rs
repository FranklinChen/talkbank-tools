//! Source-bound literal quotient. Removing a word with no possible counterpart
//! removes only zero-score consumption. Turn/episode boundaries remain, and
//! every projected path lifts by reinserting those words in their chain order.

use super::*;
use std::collections::HashSet;

/// Owns both original/projected layouts and their order-preserving maps.
/// No projected proof or mutable mapping escapes this observation producer.
pub(super) struct SearchProjection<'source> {
    original: Vec<SpeakerTurn<'source>>,
    reference: &'source [String],
    addresses: Vec<Address>,
    projected: Vec<Vec<String>>,
    projected_reference: Vec<String>,
    source_map: Vec<usize>,
    reference_map: Vec<usize>,
    mode: MatchMode,
}

impl<'source> SearchProjection<'source> {
    pub(super) fn admit<C: WorkControl + ?Sized>(
        turns: &[SpeakerTurn<'source>],
        reference: &'source [String],
        mode: MatchMode,
        control: &C,
    ) -> Result<Result<Self, LocalInterleavingRefusal>, C::Error> {
        control.check()?;
        // Even entirely empty projected turns need their boundary nodes.
        // Refuse this shape before allocating either layout or lifting maps.
        if turns.len().saturating_add(1) > SCORE_CELLS / 6 {
            return Ok(Err(LocalInterleavingRefusal::MemoryBudgetExceeded));
        }
        let source_len = turns
            .iter()
            .fold(0usize, |len, turn| len.saturating_add(turn.words.len()));
        let input = source_len.max(reference.len());
        if input > WORK_CELLS {
            return Ok(Err(LocalInterleavingRefusal::WorkBudgetExceeded {
                phase: LocalInterleavingWorkPhase::FeasibleWitness,
                required_at_least: input,
            }));
        }
        let literal = !matches!(mode, MatchMode::Fuzzy { .. });
        // This is exactly the existing equality relation, not Unicode folding
        // or fuzzy equivalence. Fuzzy observations retain the identity layout.
        let key = |word: &str| match mode {
            MatchMode::Exact => word.to_owned(),
            _ => word.to_ascii_lowercase(),
        };
        let mut reference_keys = HashSet::new();
        if literal {
            for (index, word) in reference.iter().enumerate() {
                if index % 1024 == 0 {
                    control.check()?;
                }
                reference_keys.insert(key(word));
            }
        }
        let mut source_keys = HashSet::new();
        let mut addresses = Vec::with_capacity(source_len);
        let mut projected = Vec::with_capacity(turns.len());
        let mut source_map = Vec::new();
        for (utterance, turn) in turns.iter().enumerate() {
            let mut retained = Vec::new();
            for (word, text) in turn.words.iter().enumerate() {
                let index = addresses.len();
                if index % 1024 == 0 {
                    control.check()?;
                }
                addresses.push(Address { utterance, word });
                if !literal || reference_keys.contains(&key(text)) {
                    retained.push(text.clone());
                    source_map.push(index);
                    if literal {
                        source_keys.insert(key(text));
                    }
                }
            }
            projected.push(retained);
        }
        let mut projected_reference = Vec::new();
        let mut reference_map = Vec::new();
        for (index, word) in reference.iter().enumerate() {
            if index % 1024 == 0 {
                control.check()?;
            }
            if !literal || source_keys.contains(&key(word)) {
                projected_reference.push(word.clone());
                reference_map.push(index);
            }
        }
        Ok(Ok(Self {
            original: turns.to_vec(),
            reference,
            addresses,
            projected,
            projected_reference,
            source_map,
            reference_map,
            mode,
        }))
    }

    pub(super) fn observe<C: WorkControl + ?Sized>(
        self,
        control: &C,
    ) -> Result<Result<LocalInterleaving<'source>, LocalInterleavingRefusal>, C::Error> {
        // Keep empty turns and every continuation: dropping them can authorize
        // different episodes or erase a same-speaker ordering boundary.
        let turns: Vec<_> = self
            .original
            .iter()
            .zip(&self.projected)
            .map(|(turn, words)| SpeakerTurn {
                speaker: turn.speaker,
                words,
                continuation: turn.continuation,
            })
            .collect();
        let nodes = match Budget::nodes(&turns) {
            Ok(nodes) => nodes,
            Err(refusal) => return Ok(Err(refusal)),
        };
        let (graph, projected_addresses) = Graph::of(&turns, nodes);
        control.check()?;
        let source = projected_addresses
            .iter()
            .map(|a| turns[a.utterance].words[a.word].as_str())
            .collect();
        let reference = &self.projected_reference;
        let relation = Relation::new(source, reference, self.mode);
        let (evidence, selected) = match Band::admit(
            &graph,
            &relation,
            projected_addresses.len(),
            reference.len(),
            control,
        )? {
            Ok(band) => {
                let forward = match ForwardScores::observe(&band, control)? {
                    Ok(forward) => forward,
                    Err(refusal) => return Ok(Err(refusal)),
                };
                let optimum = forward.last().get(graph.rank[graph.end]);
                let mut evidence: Vec<_> = projected_addresses
                    .iter()
                    .map(|_| WordEvidence {
                        candidates: Vec::new(),
                        can_be_missing: false,
                    })
                    .collect();
                let mut backward = band.terminal();
                band.observe_edges(forward.last(), &backward, None, optimum, &mut evidence);
                let mut candidate_count = 0usize;
                let mut end = reference.len();
                // Exactly the admitted graph proof, not a second monotonic alignment.
                while end > 0 {
                    let block = forward.block_ending_at(end, control)?;
                    for (k, prefix) in block.steps_rev() {
                        control.check()?;
                        let next = backward;
                        backward = band.backward(&next, k);
                        candidate_count += band.observe_edges(
                            prefix,
                            &backward,
                            Some((&next, k)),
                            optimum,
                            &mut evidence,
                        );
                        if candidate_count > CANDIDATES {
                            return Ok(Err(LocalInterleavingRefusal::CandidateBudgetExceeded));
                        }
                    }
                    end = block.start();
                }
                (evidence, forward.traceback(control)?)
            }
            Err(refused) => {
                let observed = match packed::observe(refused, control)? {
                    Ok(observed) => observed,
                    Err(refusal) => return Ok(Err(refusal)),
                };
                (observed.words, observed.selected)
            }
        };
        let selected = selected
            .into_iter()
            .map(|matched| Selected {
                source: self.source_map[matched.source],
                reference: self.reference_map[matched.reference],
            })
            .collect();
        // Omitted words are provably never matched. Lift candidates through
        // this owner's reference layout before exposing any common capability.
        let mut words: Vec<_> = self
            .addresses
            .iter()
            .map(|_| WordEvidence {
                candidates: Vec::new(),
                can_be_missing: true,
            })
            .collect();
        for (&original, mut observed) in self.source_map.iter().zip(evidence) {
            observed.candidates.reverse();
            for reference in &mut observed.candidates {
                *reference = self.reference_map[*reference];
            }
            words[original] = observed;
        }
        let required = words
            .iter()
            .enumerate()
            .filter_map(|(index, evidence)| {
                (!evidence.can_be_missing && !evidence.candidates.is_empty()).then_some(index)
            })
            .collect();
        let search_frontiers = search_frontiers(&self.original)
            .into_iter()
            .map(|frontier| SearchFrontier {
                before: self
                    .addresses
                    .partition_point(|a| a.utterance < frontier.before),
                after: self
                    .addresses
                    .partition_point(|a| a.utterance < frontier.after),
            })
            .collect();
        Ok(Ok(LocalInterleaving {
            turns: self.original,
            reference: self.reference,
            addresses: self.addresses,
            words,
            selected,
            search_frontiers,
            required,
        }))
    }
}
