//! Consuming graph proof; pending frontiers never escape before completion.

use super::*;

impl MaterializedGraph<'_, '_> {
    pub(in crate::dp_align::interleaving::episodes) fn observe<C: WorkControl + ?Sized>(
        self,
        reference: &[String],
        mode: MatchMode,
        control: &C,
    ) -> Result<Result<CompletedGraphObservation, LocalInterleavingRefusal>, C::Error> {
        let graph = &self.graph;
        let projected_addresses = &self.addresses;
        let source = projected_addresses
            .iter()
            .map(|a| self.turns[a.utterance].words[a.word].as_str())
            .collect();
        let relation = Relation::new(source, reference, mode);
        let (evidence, selected) = match Band::admit(
            graph,
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
                            return Ok(Err(LocalInterleavingRefusal::Budget(
                                LocalInterleavingBudget::CandidateBudgetExceeded,
                            )));
                        }
                    }
                    end = block.start();
                }
                match forward.traceback(control)? {
                    Ok(selected) => (evidence, selected),
                    Err(inconsistency) => {
                        return Ok(Err(LocalInterleavingRefusal::Inconsistent(inconsistency)));
                    }
                }
            }
            Err(BandAlternative::Refused(cause)) => return Ok(Err(cause)),
            Err(BandAlternative::Packed(prepared)) => {
                let observed = match packed::observe(prepared, control)? {
                    Ok(observed) => observed,
                    Err(refusal) => return Ok(Err(refusal)),
                };
                (observed.words, observed.selected)
            }
        };
        let frontiers = self.frontiers.finish(control)?;
        Ok(Ok(CompletedGraphObservation {
            data: GraphObservationData {
                words: evidence,
                selected,
                frontiers,
            },
        }))
    }
}
