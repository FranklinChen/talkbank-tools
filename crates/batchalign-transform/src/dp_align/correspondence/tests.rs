use super::*;
use std::collections::BTreeSet;

fn words(text: &str) -> Vec<String> {
    text.split_whitespace().map(str::to_owned).collect()
}

fn common(payload: &[String], reference: &[String], mode: MatchMode) -> BTreeSet<(usize, usize)> {
    match admit(payload, reference, mode, MAX_EDGES, MAX_FUZZY_COMPARISONS) {
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
        observation.selected()[0],
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
