use super::*;
use std::collections::BTreeSet;

fn words(text: &str) -> Vec<String> {
    text.split_whitespace().map(str::to_owned).collect()
}

fn common(payload: &[String], reference: &[String], mode: MatchMode) -> BTreeSet<(usize, usize)> {
    match CorrespondenceAdmission::observe(payload, reference, mode) {
        CorrespondenceAdmission::Complete(common) => common.pairs().collect(),
        CorrespondenceAdmission::BudgetExhausted(_) => panic!("tiny control must be admitted"),
    }
}

/// Independent exhaustive chain enumeration; intentionally only tiny inputs.
fn oracle(payload: &[String], reference: &[String], mode: MatchMode) -> BTreeSet<(usize, usize)> {
    fn enumerate(
        payload: &[String],
        reference: &[String],
        mode: MatchMode,
        p: usize,
        r: usize,
        chain: &mut Vec<(usize, usize)>,
        chains: &mut Vec<Vec<(usize, usize)>>,
    ) {
        chains.push(chain.clone());
        for i in p..payload.len() {
            for j in r..reference.len() {
                let matches = match mode {
                    MatchMode::Exact => payload[i] == reference[j],
                    MatchMode::CaseInsensitive => payload[i].eq_ignore_ascii_case(&reference[j]),
                    MatchMode::Fuzzy { threshold } => PreparedFuzzyWord::new(&payload[i]).matches(
                        &PreparedFuzzyWord::new(&reference[j]),
                        FuzzyComparison::new(threshold),
                    ),
                };
                if matches {
                    chain.push((i, j));
                    enumerate(payload, reference, mode, i + 1, j + 1, chain, chains);
                    chain.pop();
                }
            }
        }
    }
    let mut chains = Vec::new();
    enumerate(payload, reference, mode, 0, 0, &mut Vec::new(), &mut chains);
    let longest = chains
        .iter()
        .map(Vec::len)
        .max()
        .expect("empty chain exists");
    let mut optimal = chains.into_iter().filter(|chain| chain.len() == longest);
    let mut intersection: BTreeSet<_> = optimal
        .next()
        .expect("optimum exists")
        .into_iter()
        .collect();
    for chain in optimal {
        let chain: BTreeSet<_> = chain.into_iter().collect();
        intersection.retain(|pair| chain.contains(pair));
    }
    intersection
}

#[test]
fn correspondence_sparse_ranks_equal_every_optimal_chain_intersection() {
    let mut sequences = vec![Vec::new()];
    for length in 1..=4 {
        for bits in 0..(1usize << length) {
            sequences.push(
                (0..length)
                    .map(|i| if bits & (1 << i) == 0 { "a" } else { "b" }.to_owned())
                    .collect(),
            );
        }
    }
    for payload in &sequences {
        for reference in &sequences {
            assert_eq!(
                common(payload, reference, MatchMode::Exact),
                oracle(payload, reference, MatchMode::Exact),
                "payload={payload:?} reference={reference:?}"
            );
            let observation = CorrespondenceAnalysis::observe(payload, reference, MatchMode::Exact);
            let selected: BTreeSet<_> = observation
                .selected()
                .expect("tiny inspection completes")
                .iter()
                .filter_map(|item| match item {
                    AlignResult::Match {
                        payload_idx,
                        reference_idx,
                        ..
                    } => Some((*payload_idx, *reference_idx)),
                    _ => None,
                })
                .collect();
            assert!(common(payload, reference, MatchMode::Exact).is_subset(&selected));
        }
    }
}

#[test]
fn correspondence_fuzzy_and_ascii_policies_share_the_alignment_relation() {
    let controls = [
        ("gonna now", "gona now"),
        ("A à", "a À à"),
        ("yeah now yeah", "yea now yeah yeah"),
        ("hello", "hello hello"),
        ("aaaa aaab aabb aaaa", "aaab aabb aaaa aaab"),
    ];
    for (payload, reference) in controls {
        for mode in [
            MatchMode::Exact,
            MatchMode::CaseInsensitive,
            MatchMode::Fuzzy { threshold: 0.85 },
            MatchMode::Fuzzy { threshold: 0.80 },
        ] {
            assert_eq!(
                common(&words(payload), &words(reference), mode),
                oracle(&words(payload), &words(reference), mode)
            );
        }
    }
    // Every word pair is dissimilar, so the complete relation is empty and
    // retains no candidate edges. Its 64,008,000 occurrence pairs are not a
    // storage demand or a linguistic reason to refuse proof. Exact spelling
    // classes need only one lexical comparison for this population.
    let payload = vec!["a".to_owned(); 8000];
    let reference = vec!["zzzz".to_owned(); 8001];
    assert!(matches!(
        CorrespondenceAdmission::observe(&payload, &reference, MatchMode::Fuzzy { threshold: 0.85 }),
        CorrespondenceAdmission::Complete(common) if common.pairs().next().is_none()
    ));
}

#[test]
fn correspondence_identical_selected_prefix_does_not_prove_uniqueness() {
    let observation = CorrespondenceAnalysis::observe(
        &words("hello now"),
        &words("hello hello now"),
        MatchMode::Exact,
    );
    assert!(matches!(
        observation.selected().expect("tiny inspection completes")[0],
        AlignResult::Match {
            payload_idx: 0,
            reference_idx: 0,
            ..
        }
    ));
    assert_eq!(
        common(
            &words("hello now"),
            &words("hello hello now"),
            MatchMode::Exact
        ),
        BTreeSet::from([(1, 2)])
    );
}

#[test]
fn correspondence_budget_exhaustion_is_not_ambiguity_or_empty_proof() {
    let payload = words("a a");
    let reference = words("a a a");
    for mode in [MatchMode::Exact, MatchMode::Fuzzy { threshold: 0.85 }] {
        assert!(matches!(
            admit(&payload, &reference, mode, 5),
            CorrespondenceAdmission::BudgetExhausted(budget) if budget.limit() == 5
        ));
    }
    assert!(matches!(
        admit(&[], &reference, MatchMode::Exact, 0),
        CorrespondenceAdmission::Complete(_)
    ));
}

#[test]
fn repeated_literal_words_do_not_require_retaining_nonoptimal_edges() {
    for length in [1300, 5000] {
        let mut source = vec!["same".to_owned(); length];
        let mut reference = source.clone();
        // One disagreement defeats whole-input exact embedding. Every other
        // pair remains common, despite the quadratic raw equality population.
        let disagreement = length / 2;
        source[disagreement] = "source".to_owned();
        reference[disagreement] = "reference".to_owned();
        let observed = CorrespondenceAdmission::observe(&source, &reference, MatchMode::Exact);
        let CorrespondenceAdmission::Complete(common) = observed else {
            panic!(
                "{length} words have only {} optimal edges, not the raw quadratic population",
                length - 1
            );
        };
        assert_eq!(
            common.pairs().collect::<Vec<_>>(),
            (0..length)
                .filter(|&i| i != disagreement)
                .map(|i| (i, i))
                .collect::<Vec<_>>()
        );
    }
}

fn packed_common(
    payload: &[String],
    reference: &[String],
    mode: super::super::LiteralMatchMode,
) -> BTreeSet<(usize, usize)> {
    let input =
        literal::LiteralCandidates::observe(payload, reference, mode, &Uninterrupted).unwrap();
    match input.certify(MAX_EDGES, &Uninterrupted).unwrap() {
        Some(CorrespondenceAdmission::Complete(common)) => common.pairs().collect(),
        _ => panic!("small packed control must complete"),
    }
}

#[test]
fn packed_frontier_common_pairs_equal_exhaustive_chain_intersections() {
    let mut sequences = vec![Vec::new()];
    for len in 1..=4 {
        for bits in 0..1usize << len {
            sequences.push(
                (0..len)
                    .map(|i| if bits & (1 << i) == 0 { "a" } else { "b" }.to_owned())
                    .collect(),
            );
        }
    }
    for payload in &sequences {
        for reference in &sequences {
            assert_eq!(
                packed_common(payload, reference, super::super::LiteralMatchMode::Exact),
                oracle(payload, reference, MatchMode::Exact)
            );
        }
    }
    for (payload, reference) in [
        ("A é", "a É é"),
        ("head repeat tail", "head repeat repeat tail"),
    ] {
        for (mode, literal_mode) in [
            (MatchMode::Exact, super::super::LiteralMatchMode::Exact),
            (
                MatchMode::CaseInsensitive,
                super::super::LiteralMatchMode::AsciiInsensitive,
            ),
        ] {
            assert_eq!(
                packed_common(&words(payload), &words(reference), literal_mode),
                oracle(&words(payload), &words(reference), mode)
            );
        }
    }
}

#[test]
fn packed_literal_admission_preserves_true_competition_and_edge_overflow() {
    let payload = vec!["same".to_owned(); 1300];
    let reference = vec!["same".to_owned(); 1301];
    assert!(common(&payload, &reference, MatchMode::Exact).is_empty());
    let payload = vec!["same".to_owned(); 1000];
    let reference = vec!["same".to_owned(); 3000];
    assert!(matches!(
        admit(
            &payload,
            &reference,
            MatchMode::Exact,
            MAX_EDGES
        ),
        CorrespondenceAdmission::BudgetExhausted(budget) if budget.limit() == MAX_EDGES
    ));
}

#[test]
fn packed_workspace_admission_and_interruption_do_not_create_common_pairs() {
    let payload = vec!["same".to_owned(); 50];
    let reference = vec!["same".to_owned(); 400_000];
    let input = literal::LiteralCandidates::observe(
        &payload,
        &reference,
        super::super::LiteralMatchMode::Exact,
        &Uninterrupted,
    )
    .unwrap();
    assert!(input.certify(MAX_EDGES, &Uninterrupted).unwrap().is_none());

    struct Count(std::cell::Cell<usize>);
    impl WorkControl for Count {
        type Error = std::convert::Infallible;
        fn check(&self) -> Result<(), Self::Error> {
            self.0.set(self.0.get() + 1);
            Ok(())
        }
    }
    struct InterruptAfter(std::cell::Cell<usize>);
    impl WorkControl for InterruptAfter {
        type Error = &'static str;
        fn check(&self) -> Result<(), Self::Error> {
            match self.0.get().checked_sub(1) {
                Some(remaining) => {
                    self.0.set(remaining);
                    Ok(())
                }
                None => Err("cancelled"),
            }
        }
    }
    let payload = words("a a");
    let reference = words("a a a");
    let input = literal::LiteralCandidates::observe(
        &payload,
        &reference,
        super::super::LiteralMatchMode::Exact,
        &Uninterrupted,
    )
    .unwrap();
    let count = Count(std::cell::Cell::new(0));
    assert!(matches!(
        input.certify(MAX_EDGES, &count).unwrap(),
        Some(CorrespondenceAdmission::Complete(_))
    ));
    for at in 0..count.0.get() {
        assert!(
            matches!(
                input.certify(MAX_EDGES, &InterruptAfter(std::cell::Cell::new(at))),
                Err("cancelled")
            ),
            "at checkpoint {at}"
        );
    }
}

#[test]
fn packed_common_evidence_snapshot_distinguishes_competing_occurrences() {
    let snapshot = [
        ("a a", "a a a"),
        ("head repeat tail", "head repeat repeat tail"),
        ("A é", "a É é"),
    ]
    .into_iter()
    .map(|(source, reference)| {
        format!(
            "{source} / {reference} => {:?}",
            packed_common(
                &words(source),
                &words(reference),
                super::super::LiteralMatchMode::AsciiInsensitive
            )
        )
    })
    .collect::<Vec<_>>()
    .join("\n");
    insta::assert_snapshot!(snapshot, @"
    a a / a a a => {}
    head repeat tail / head repeat repeat tail => {(0, 0), (2, 3)}
    A é / a É é => {(0, 0), (1, 2)}
    ");
}

#[test]
fn interruption_after_partial_rank_observation_never_emits_a_certificate() {
    enum CheckMode {
        Observe,
        InterruptAt(usize),
    }
    struct Checks {
        count: std::cell::Cell<usize>,
        mode: CheckMode,
    }
    impl WorkControl for Checks {
        type Error = &'static str;
        fn check(&self) -> Result<(), Self::Error> {
            let next = self.count.get() + 1;
            self.count.set(next);
            match self.mode {
                CheckMode::InterruptAt(stop) if next >= stop => Err("cancelled"),
                CheckMode::Observe | CheckMode::InterruptAt(_) => Ok(()),
            }
        }
    }
    let mut payload = vec!["same".to_owned(); 64];
    payload.push("tail".into());
    let mut reference = vec!["same".to_owned(); 65];
    reference.push("tail".into());
    let input = literal::LiteralCandidates::observe(
        &payload,
        &reference,
        super::super::LiteralMatchMode::Exact,
        &Uninterrupted,
    )
    .unwrap();
    let completed = Checks {
        count: std::cell::Cell::new(0),
        mode: CheckMode::Observe,
    };
    let Some(CorrespondenceAdmission::Complete(common)) =
        input.certify(MAX_EDGES, &completed).unwrap()
    else {
        panic!("bounded control completes");
    };
    assert_eq!(common.pairs().collect::<Vec<_>>(), vec![(64, 65)]);
    let interrupted = Checks {
        count: std::cell::Cell::new(0),
        mode: CheckMode::InterruptAt(completed.count.get() - 8),
    };
    assert!(matches!(
        input.certify(MAX_EDGES, &interrupted),
        Err("cancelled")
    ));
    // Exercise cancellation throughout preparation, completed-row reuse and
    // both sparse score phases through the real public producer.
    let payload = words("aaaa aaab aabb aaaa");
    let reference = words("aaab aabb aaaa aaab");
    for mode in [MatchMode::Exact, MatchMode::Fuzzy { threshold: 0.80 }] {
        let completed = Checks {
            count: std::cell::Cell::new(0),
            mode: CheckMode::Observe,
        };
        assert!(matches!(
            CorrespondenceAdmission::observe_controlled(&payload, &reference, mode, &completed),
            Ok(CorrespondenceAdmission::Complete(_))
        ));
        for at in 1..=completed.count.get() {
            let interrupted = Checks {
                count: std::cell::Cell::new(0),
                mode: CheckMode::InterruptAt(at),
            };
            assert!(
                matches!(
                    CorrespondenceAdmission::observe_controlled(
                        &payload,
                        &reference,
                        mode,
                        &interrupted
                    ),
                    Err("cancelled")
                ),
                "mode={mode:?} checkpoint={at}"
            );
        }
    }
}

#[test]
fn correspondence_long_sparse_deletion_preserves_all_remaining_recovery() {
    let payload: Vec<_> = (0..3000).map(|i| format!("word{i}")).collect();
    let reference: Vec<_> = payload
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != 1499)
        .map(|(_, word)| word.clone())
        .collect();
    let admitted = common(&payload, &reference, MatchMode::CaseInsensitive);
    assert_eq!(admitted.len(), 2999);
    assert!(admitted.contains(&(1498, 1498)));
    assert!(admitted.contains(&(1500, 1499)));
}
