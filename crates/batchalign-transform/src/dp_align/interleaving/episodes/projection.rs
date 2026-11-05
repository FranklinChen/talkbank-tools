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
            return Ok(Err(LocalInterleavingRefusal::Budget(
                LocalInterleavingBudget::MemoryBudgetExceeded,
            )));
        }
        let source_len = turns
            .iter()
            .fold(0usize, |len, turn| len.saturating_add(turn.words.len()));
        let input = source_len.max(reference.len());
        if input > WORK_CELLS {
            return Ok(Err(LocalInterleavingRefusal::Budget(
                LocalInterleavingBudget::WorkBudgetExceeded {
                    phase: LocalInterleavingWorkPhase::FeasibleWitness,
                    required_at_least: input,
                },
            )));
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
        let prepared = match PreparedGraph::admit(&turns) {
            Ok(prepared) => prepared,
            Err(refusal) => return Ok(Err(refusal)),
        };
        let graph = prepared.materialize(control)?;
        let observed = match graph.observe(&self.projected_reference, self.mode, control)? {
            Ok(observed) => observed,
            Err(refusal) => return Ok(Err(refusal)),
        };
        drop(turns);
        self.lift(observed, control)
    }

    fn lift<C: WorkControl + ?Sized>(
        self,
        observed: CompletedGraphObservation,
        control: &C,
    ) -> Result<Result<LocalInterleaving<'source>, LocalInterleavingRefusal>, C::Error> {
        control.check()?;
        let data = observed.into_data();
        let evidence = data.words;
        let selected = data.selected;
        let frontiers = data.frontiers;
        let Self {
            original,
            reference,
            addresses,
            projected,
            projected_reference,
            source_map,
            reference_map,
            mode: _,
        } = self;
        drop(projected);
        drop(projected_reference);
        let selected = selected
            .into_iter()
            .map(|matched| Selected {
                source: source_map[matched.source],
                reference: reference_map[matched.reference],
            })
            .collect();
        // Omitted words are provably never matched. Lift candidates through
        // this owner's reference layout before exposing any common capability.
        let mut words: Vec<_> = addresses
            .iter()
            .map(|_| WordEvidence {
                candidates: Vec::new(),
                can_be_missing: true,
            })
            .collect();
        for (&original, mut observed) in source_map.iter().zip(evidence) {
            observed.candidates.reverse();
            for reference in &mut observed.candidates {
                *reference = reference_map[*reference];
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
        let search_frontiers = frontiers
            .into_iter()
            .map(|frontier| SearchFrontier {
                before: addresses.partition_point(|a| a.utterance < frontier.before),
                after: addresses.partition_point(|a| a.utterance < frontier.after),
            })
            .collect();
        Ok(Ok(LocalInterleaving {
            turns: original,
            reference,
            addresses,
            words,
            selected,
            search_frontiers,
            required,
        }))
    }
}
