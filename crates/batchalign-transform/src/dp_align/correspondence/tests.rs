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
    ];
    for (payload, reference) in controls {
        for mode in [
            MatchMode::Exact,
            MatchMode::CaseInsensitive,
            MatchMode::Fuzzy { threshold: 0.85 },
        ] {
            assert_eq!(
                common(&words(payload), &words(reference), mode),
                oracle(&words(payload), &words(reference), mode)
            );
        }
    }
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
    assert!(matches!(
        admit(&payload, &reference, MatchMode::Exact, 5, 100),
        CorrespondenceAdmission::BudgetExhausted(CorrespondenceBudget::CandidateEdges)
    ));
    assert!(matches!(
        admit(
            &payload,
            &reference,
            MatchMode::Fuzzy { threshold: 0.85 },
            100,
            5
        ),
        CorrespondenceAdmission::BudgetExhausted(CorrespondenceBudget::FuzzyComparisons)
    ));
    assert!(matches!(
        admit(&[], &reference, MatchMode::Exact, 0, 0),
        CorrespondenceAdmission::Complete(_)
    ));
}

#[test]
fn repeated_literal_words_do_not_require_retaining_nonoptimal_edges() {
    let source = vec!["same".to_owned(); 1300];
    let observed = admit(
        &source,
        &source,
        MatchMode::Exact,
        MAX_EDGES,
        MAX_FUZZY_COMPARISONS,
    );
    let CorrespondenceAdmission::Complete(common) = observed else {
        panic!("1,690,000 raw equal-word pairs have only 1,300 optimal edges");
    };
    assert_eq!(
        common.pairs().collect::<Vec<_>>(),
        (0..1300).map(|i| (i, i)).collect::<Vec<_>>()
    );
}

fn packed_common(
    payload: &[String],
    reference: &[String],
    mode: MatchMode,
) -> BTreeSet<(usize, usize)> {
    let mode = crate::dp_align::packed_literal::policy(mode).unwrap();
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
                packed_common(payload, reference, MatchMode::Exact),
                oracle(payload, reference, MatchMode::Exact)
            );
        }
    }
    for (payload, reference) in [
        ("A é", "a É é"),
        ("head repeat tail", "head repeat repeat tail"),
    ] {
        for mode in [MatchMode::Exact, MatchMode::CaseInsensitive] {
            assert_eq!(
                packed_common(&words(payload), &words(reference), mode),
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
            MAX_EDGES,
            MAX_FUZZY_COMPARISONS
        ),
        CorrespondenceAdmission::BudgetExhausted(CorrespondenceBudget::CandidateEdges)
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

    struct Cancel;
    impl WorkControl for Cancel {
        type Error = &'static str;
        fn check(&self) -> Result<(), Self::Error> {
            Err("cancelled")
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
    assert!(matches!(
        input.certify(MAX_EDGES, &Cancel),
        Err("cancelled")
    ));
    assert!(
        crate::dp_align::packed_literal::policy(MatchMode::Fuzzy { threshold: 0.85 }).is_none()
    );
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
                MatchMode::CaseInsensitive
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
    struct Checks {
        count: std::cell::Cell<usize>,
        stop: usize,
    }
    impl WorkControl for Checks {
        type Error = &'static str;
        fn check(&self) -> Result<(), Self::Error> {
            let next = self.count.get() + 1;
            self.count.set(next);
            if next >= self.stop {
                Err("cancelled")
            } else {
                Ok(())
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
        stop: usize::MAX,
    };
    let Some(CorrespondenceAdmission::Complete(common)) =
        input.certify(MAX_EDGES, &completed).unwrap()
    else {
        panic!("bounded control completes");
    };
    assert_eq!(common.pairs().collect::<Vec<_>>(), vec![(64, 65)]);
    let interrupted = Checks {
        count: std::cell::Cell::new(0),
        stop: completed.count.get() - 8,
    };
    assert!(matches!(
        input.certify(MAX_EDGES, &interrupted),
        Err("cancelled")
    ));
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
