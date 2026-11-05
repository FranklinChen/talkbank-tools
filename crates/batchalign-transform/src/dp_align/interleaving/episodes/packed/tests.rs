use super::*;
use std::collections::BTreeSet;
use std::num::NonZeroUsize;

fn words(bits: usize, len: usize) -> Vec<String> {
    (0..len)
        .map(|i| if bits & (1 << i) == 0 { "a" } else { "b" }.to_owned())
        .collect()
}

fn compare(turns: &[SpeakerTurn<'_>], reference: &[String]) {
    let materialized = PreparedGraph::admit(turns)
        .unwrap()
        .materialize(&Uninterrupted)
        .unwrap();
    let graph = materialized.graph();
    let addresses = materialized.addresses();
    let relation = Relation::new(
        addresses
            .iter()
            .map(|address| turns[address.utterance].words[address.word].as_str())
            .collect(),
        reference,
        MatchMode::Exact,
    );
    let expected = super::super::tests::oracle(turns, reference);
    let optimum = expected[0].len();
    for block in [1, 2, 3, graph.order.len()] {
        let block = NonZeroUsize::new(block).expect("declared fixture strides are positive");
        let result = uninterrupted(kernel::testing_full(
            PreparedLiteralSearch::testing(graph, &relation),
            block,
            &Uninterrupted,
        ))
        .unwrap();
        let prepared = PreparedLiteralSearch::testing(graph, &relation);
        let mut results = vec![result];
        for problem in prepared.testing_bounds() {
            results.push(uninterrupted(kernel::testing(problem, block, &Uninterrupted)).unwrap());
        }
        for result in results {
            let selected: BTreeSet<_> = result
                .selected
                .iter()
                .map(|matched| {
                    let address = addresses[matched.source];
                    (address.utterance, address.word, matched.reference)
                })
                .collect();
            assert_eq!(selected.len(), optimum);
            assert!(expected.contains(&selected));
            for (index, word) in result.words.iter().enumerate() {
                let address = addresses[index];
                let candidates: BTreeSet<_> = expected
                    .iter()
                    .flat_map(|path| path.iter())
                    .filter(|&&(u, w, _)| u == address.utterance && w == address.word)
                    .map(|&(_, _, r)| r)
                    .collect();
                assert_eq!(
                    word.candidates.iter().copied().collect::<BTreeSet<_>>(),
                    candidates
                );
                assert_eq!(
                    word.can_be_missing,
                    expected.iter().any(|path| path
                        .iter()
                        .all(|&(u, w, _)| u != address.utterance || w != address.word))
                );
            }
        }
    }
}

#[test]
fn full_profile_inspection_selection_is_checkpoint_independent() {
    let a = words(2, 2);
    let b = words(0, 2);
    let turns = [SpeakerTurn::new("A", &a), SpeakerTurn::new("B", &b)];
    let mut selections = Vec::new();
    for bits in [0, 9, 17] {
        let reference = words(bits, 5);
        compare(&turns, &reference);
        let materialized = PreparedGraph::admit(&turns)
            .unwrap()
            .materialize(&Uninterrupted)
            .unwrap();
        let graph = materialized.graph();
        let addresses = materialized.addresses();
        let relation = Relation::new(
            addresses
                .iter()
                .map(|p| turns[p.utterance].words[p.word].as_str())
                .collect(),
            &reference,
            MatchMode::Exact,
        );
        let mut selected = None;
        for block in [1, 2, graph.order.len()] {
            let block = NonZeroUsize::new(block).expect("declared fixture strides are positive");
            let observed = uninterrupted(kernel::testing_full(
                PreparedLiteralSearch::testing(graph, &relation),
                block,
                &Uninterrupted,
            ))
            .unwrap();
            let pairs: Vec<_> = observed
                .selected
                .iter()
                .map(|pair| format!("{}:{}", pair.source, pair.reference))
                .collect();
            if let Some(prior) = &selected {
                assert_eq!(prior, &pairs);
            }
            selected = Some(pairs);
        }
        selections.push(selected.unwrap());
    }
    // Compatibility observation from the old producer, not a correctness gold:
    // the independent legal-order oracle above establishes proof semantics.
    insta::assert_debug_snapshot!(selections);
}

#[test]
fn packed_joint_proofs_equal_independent_legal_order_assignments_across_blocks() {
    for m in 0..=3 {
        for n in 0..=4 {
            for a in 0..1usize << m {
                for b in 0..1usize << n {
                    let source = words(a, m);
                    let reference = words(b, n);
                    for split in 0..=m {
                        compare(
                            &[
                                SpeakerTurn::new("A", &source[..split]),
                                SpeakerTurn::new("B", &source[split..]),
                            ],
                            &reference,
                        );
                    }
                }
            }
        }
    }
    let a = words(0, 1);
    let b = words(1, 1);
    let c = words(0, 1);
    for bits in 0..16 {
        let reference = words(bits, 4);
        compare(
            &[
                SpeakerTurn::new("A", &a),
                SpeakerTurn::new("B", &b),
                SpeakerTurn::overlapping_continuation("B", &c),
            ],
            &reference,
        );
        compare(
            &[
                SpeakerTurn::new("A", &a),
                SpeakerTurn::new("B", &b),
                SpeakerTurn::new("B", &c),
            ],
            &reference,
        );
    }
}

#[test]
fn a_work_refused_reordered_dialogue_uses_the_packed_producer() {
    let a = vec!["a".to_owned(); 380];
    let b = vec!["b".to_owned(); 380];
    let reference: Vec<_> = b.iter().chain(&a).cloned().collect();
    let turns = [SpeakerTurn::new("A", &a), SpeakerTurn::new("B", &b)];
    let materialized = PreparedGraph::admit(&turns)
        .unwrap()
        .materialize(&Uninterrupted)
        .unwrap();
    let graph = materialized.graph();
    let addresses = materialized.addresses();
    let relation = Relation::new(
        addresses
            .iter()
            .map(|address| turns[address.utterance].words[address.word].as_str())
            .collect(),
        &reference,
        MatchMode::Exact,
    );
    let refused = match uninterrupted(Band::admit(graph, &relation, 760, 760, &Uninterrupted)) {
        Err(BandAlternative::Packed(prepared)) => prepared,
        Err(BandAlternative::Refused(cause)) => {
            panic!("fixture must prepare a literal alternative, got {cause:?}")
        }
        Ok(_) => panic!("fixture must reproduce a dense work refusal"),
    };
    let observed = uninterrupted(observe(refused, &Uninterrupted)).unwrap();
    assert_eq!(observed.selected.len(), 760);
    for (index, word) in observed.words.iter().enumerate() {
        assert!(!word.can_be_missing);
        assert_eq!(
            word.candidates,
            [if index < 380 {
                380 + index
            } else {
                index - 380
            }]
        );
    }
}

#[test]
fn cancellation_never_yields_a_complete_packed_observation() {
    use std::cell::Cell;
    struct Cancel(Cell<usize>);
    impl WorkControl for Cancel {
        type Error = &'static str;
        fn check(&self) -> Result<(), Self::Error> {
            self.0.set(self.0.get() + 1);
            if self.0.get() == 20 {
                Err("cancelled")
            } else {
                Ok(())
            }
        }
    }
    let a = words(0, 20);
    let b = words(1, 20);
    let reference = words(2, 40);
    let turns = [SpeakerTurn::new("A", &a), SpeakerTurn::new("B", &b)];
    let materialized = PreparedGraph::admit(&turns)
        .unwrap()
        .materialize(&Uninterrupted)
        .unwrap();
    let graph = materialized.graph();
    let addresses = materialized.addresses();
    let relation = Relation::new(
        addresses
            .iter()
            .map(|address| turns[address.utterance].words[address.word].as_str())
            .collect(),
        &reference,
        MatchMode::Exact,
    );
    let result = observe(
        PreparedLiteralSearch::testing(graph, &relation),
        &Cancel(Cell::new(0)),
    );
    assert!(matches!(result, Err("cancelled")));
}

#[test]
fn partial_reordering_completes_beyond_the_former_execution_work_cutoff() {
    #[derive(Default)]
    struct Recorder(std::cell::RefCell<Vec<PackedSearchCost>>);
    impl WorkControl for Recorder {
        type Error = std::convert::Infallible;
        fn check(&self) -> Result<(), Self::Error> {
            Ok(())
        }
        fn packed_search_cost(&self, cost: &PackedSearchCost) {
            self.0.borrow_mut().push(cost.clone());
        }
    }
    let a = vec!["a".to_owned(); 420];
    let b = vec!["b".to_owned(); 420];
    let reference: Vec<_> = a[..210]
        .iter()
        .chain(&b)
        .chain(&a[210..])
        .cloned()
        .collect();
    let turns = [SpeakerTurn::new("A", &a), SpeakerTurn::new("B", &b)];
    let recorder = Recorder::default();
    let observed = uninterrupted(LocalInterleaving::observe_controlled(
        &turns,
        &reference,
        MatchMode::Exact,
        &recorder,
    ))
    .unwrap();
    assert_eq!(observed.matched_words(), 840);
    for (source, word) in observed.words().enumerate() {
        let expected = match source {
            0..210 => source,
            210..420 => source + 420,
            _ => source - 210,
        };
        assert!(!word.can_be_missing());
        assert_eq!(word.common().unwrap().matched().reference_index(), expected);
    }
    let costs = recorder.0.borrow();
    assert_eq!(costs.len(), 1);
    assert_eq!(costs[0].status(), PackedSearchStatus::Completed);
    assert!(costs[0].used().lower_bound() > WORK_CELLS as u128);
}

#[test]
fn packed_profiles_preserve_exact_domains_across_shifted_word_boundaries() {
    let a = vec!["a".to_owned(); 65];
    let b = vec!["b".to_owned(); 65];
    let reference: Vec<_> = b.iter().chain(&a).cloned().collect();
    let turns = [SpeakerTurn::new("A", &a), SpeakerTurn::new("B", &b)];
    let materialized = PreparedGraph::admit(&turns)
        .unwrap()
        .materialize(&Uninterrupted)
        .unwrap();
    let graph = materialized.graph();
    let addresses = materialized.addresses();
    let relation = Relation::new(
        addresses
            .iter()
            .map(|p| turns[p.utterance].words[p.word].as_str())
            .collect(),
        &reference,
        MatchMode::Exact,
    );
    for block in [64, 129, graph.order.len()] {
        let block = NonZeroUsize::new(block).expect("declared fixture strides are positive");
        let result = uninterrupted(kernel::testing(
            PreparedLiteralSearch::testing(graph, &relation),
            block,
            &Uninterrupted,
        ))
        .unwrap();
        assert_eq!(result.selected.len(), 130);
        for (i, word) in result.words.iter().enumerate() {
            assert!(!word.can_be_missing);
            assert_eq!(word.candidates, [if i < 65 { i + 65 } else { i - 65 }]);
        }
    }

    // Closed-form all-optimal domains for repeated words cross unaligned
    // 64-slot windows, union repeated edge observations and promote inline
    // domains to sparse blocks. Enumerating all legal paths here is exponential.
    for (m, n) in [(65usize, 130usize), (130, 65)] {
        let source = vec!["same".to_owned(); m];
        let reference = vec!["same".to_owned(); n];
        let turns = [SpeakerTurn::new("A", &source)];
        let materialized = PreparedGraph::admit(&turns)
            .unwrap()
            .materialize(&Uninterrupted)
            .unwrap();
        let graph = materialized.graph();
        let relation = Relation::new(
            source.iter().map(String::as_str).collect(),
            &reference,
            MatchMode::Exact,
        );
        for block in [1, 2, graph.order.len()] {
            let block = NonZeroUsize::new(block).expect("declared fixture strides are positive");
            let results = [
                uninterrupted(kernel::testing(
                    PreparedLiteralSearch::testing(graph, &relation),
                    block,
                    &Uninterrupted,
                ))
                .unwrap(),
                uninterrupted(kernel::testing_full(
                    PreparedLiteralSearch::testing(graph, &relation),
                    block,
                    &Uninterrupted,
                ))
                .unwrap(),
            ];
            for result in results {
                assert_eq!(result.selected.len(), m.min(n));
                for (word, observed) in result.words.iter().enumerate() {
                    let first = word.saturating_sub(m.saturating_sub(n));
                    let last = (word + n.saturating_sub(m)).min(n - 1);
                    assert_eq!(
                        observed.candidates,
                        (first..=last).rev().collect::<Vec<_>>()
                    );
                    assert_eq!(observed.can_be_missing, m > n);
                }
            }
        }
    }
}

#[test]
fn interruption_inside_forward_reconstruction_proof_and_traceback_returns_no_observation() {
    use std::cell::{Cell, RefCell};
    struct Polls {
        calls: Cell<usize>,
        cancel_at: Option<usize>,
        receipts: RefCell<Vec<PackedSearchCost>>,
    }
    impl WorkControl for Polls {
        type Error = &'static str;
        fn check(&self) -> Result<(), Self::Error> {
            self.calls.set(self.calls.get() + 1);
            if self.cancel_at == Some(self.calls.get()) {
                Err("cancelled")
            } else {
                Ok(())
            }
        }
        fn packed_search_cost(&self, receipt: &PackedSearchCost) {
            self.receipts.borrow_mut().push(receipt.clone());
        }
    }
    let a = words(0, 4);
    let b = words(1, 4);
    let reference = words(2, 8);
    let turns = [SpeakerTurn::new("A", &a), SpeakerTurn::new("B", &b)];
    let materialized = PreparedGraph::admit(&turns)
        .unwrap()
        .materialize(&Uninterrupted)
        .unwrap();
    let graph = materialized.graph();
    let addresses = materialized.addresses();
    let relation = Relation::new(
        addresses
            .iter()
            .map(|address| turns[address.utterance].words[address.word].as_str())
            .collect(),
        &reference,
        MatchMode::Exact,
    );
    let mut stages = BTreeSet::new();
    let block = NonZeroUsize::new(2).expect("declared fixture stride is positive");
    for produce in [kernel::testing::<Polls>, kernel::testing_full::<Polls>] {
        let polls = Polls {
            calls: Cell::new(0),
            cancel_at: None,
            receipts: RefCell::new(Vec::new()),
        };
        assert!(
            produce(
                PreparedLiteralSearch::testing(graph, &relation),
                block,
                &polls
            )
            .unwrap()
            .is_ok()
        );
        let completed = polls.calls.get();
        for at in 1..=completed {
            let polls = Polls {
                calls: Cell::new(0),
                cancel_at: Some(at),
                receipts: RefCell::new(Vec::new()),
            };
            assert!(
                matches!(
                    produce(
                        PreparedLiteralSearch::testing(graph, &relation),
                        block,
                        &polls
                    ),
                    Err("cancelled")
                ),
                "at poll {at}"
            );
            // Admission can stop before there is a ledger. Once admitted, every
            // interruption destroys the pending proof, even after inspection has
            // visited a live block and later backward blocks remain unfinished.
            let receipts = polls.receipts.borrow();
            for cost in receipts.iter() {
                assert_eq!(cost.status(), PackedSearchStatus::Interrupted);
                stages.insert(cost.last_stage().name());
            }
        }
    }
    assert_eq!(
        stages,
        BTreeSet::from([
            "forward",
            "reconstruction",
            "backward",
            "edges",
            "traceback",
        ])
    );
}

#[test]
fn full_fallback_candidate_refusal_never_returns_partial_inspection() {
    use std::cell::Cell;
    struct Observer(Cell<usize>);
    impl WorkControl for Observer {
        type Error = std::convert::Infallible;
        fn check(&self) -> Result<(), Self::Error> {
            Ok(())
        }
        fn packed_search_cost(&self, _: &PackedSearchCost) {
            self.0.set(self.0.get() + 1);
        }
    }
    // The band needs over 69 million scalar cells; full packed reservation
    // fits, but the repeated words have far more than one million addressed
    // candidates. These are derived admission bounds, not a raised test cap.
    let source = vec!["a".to_owned(); 70_000];
    let reference = vec!["a".to_owned(); 1_000];
    let turns = [SpeakerTurn::new("A", &source)];
    let materialized = PreparedGraph::admit(&turns)
        .unwrap()
        .materialize(&Uninterrupted)
        .unwrap();
    let graph = materialized.graph();
    let addresses = materialized.addresses();
    let relation = Relation::new(
        addresses
            .iter()
            .map(|p| turns[p.utterance].words[p.word].as_str())
            .collect(),
        &reference,
        MatchMode::Exact,
    );
    let refused = match uninterrupted(Band::admit(
        graph,
        &relation,
        source.len(),
        reference.len(),
        &Uninterrupted,
    )) {
        Err(BandAlternative::Packed(prepared)) => prepared,
        Err(BandAlternative::Refused(cause)) => {
            panic!("fixture must prepare a literal alternative, got {cause:?}")
        }
        Ok(_) => panic!("the actual band must refuse this work population"),
    };
    let observer = Observer(Cell::new(0));
    assert!(matches!(
        uninterrupted(observe(refused, &observer)),
        Err(LocalInterleavingRefusal::Budget(
            LocalInterleavingBudget::CandidateBudgetExceeded
        ))
    ));
    assert_eq!(
        observer.0.get(),
        0,
        "full reservation keeps the existing observer scope"
    );
}

#[test]
fn ascii_policy_and_budget_admission_remain_explicit() {
    let a = vec!["A".to_owned(), "é".to_owned()];
    let b = vec!["b".to_owned(), "É".to_owned()];
    let reference = vec![
        "b".to_owned(),
        "a".to_owned(),
        "É".to_owned(),
        "é".to_owned(),
    ];
    let turns = [SpeakerTurn::new("A", &a), SpeakerTurn::new("B", &b)];
    let materialized = PreparedGraph::admit(&turns)
        .unwrap()
        .materialize(&Uninterrupted)
        .unwrap();
    let graph = materialized.graph();
    let addresses = materialized.addresses();
    let source: Vec<_> = addresses
        .iter()
        .map(|address| turns[address.utterance].words[address.word].as_str())
        .collect();
    let relation = Relation::new(source.clone(), &reference, MatchMode::CaseInsensitive);
    let result = uninterrupted(observe(
        PreparedLiteralSearch::testing(graph, &relation),
        &Uninterrupted,
    ))
    .unwrap();
    let lowered_a: Vec<_> = a.iter().map(|word| word.to_ascii_lowercase()).collect();
    let lowered_b: Vec<_> = b.iter().map(|word| word.to_ascii_lowercase()).collect();
    let expected = super::super::tests::oracle(
        &[
            SpeakerTurn::new("A", &lowered_a),
            SpeakerTurn::new("B", &lowered_b),
        ],
        &reference,
    );
    for (index, word) in result.words.iter().enumerate() {
        let address = addresses[index];
        let candidates: BTreeSet<_> = expected
            .iter()
            .flat_map(|path| path.iter())
            .filter(|&&(u, w, _)| u == address.utterance && w == address.word)
            .map(|&(_, _, r)| r)
            .collect();
        assert_eq!(
            word.candidates.iter().copied().collect::<BTreeSet<_>>(),
            candidates
        );
    }
    // Admission consumes this real reference population; no fixture can pair
    // independent dimensions with a prepared relation or replace its geometry.
    let reference = vec!["literal".to_owned(); SCORE_CELLS / 24 + 1];
    let relation = Relation::new(source, &reference, MatchMode::CaseInsensitive);
    let prepared = uninterrupted(geometry::SurveyedLiteralSearch::observe(
        PreparedLiteralSearch::testing(graph, &relation),
        &Uninterrupted,
    ))
    .expect("the graph survey fits");
    let memory = prepared.project(Representation::Full);
    assert!(matches!(
        memory,
        Err(LocalInterleavingRefusal::Budget(
            LocalInterleavingBudget::MemoryBudgetExceeded
        ))
    ));
}
