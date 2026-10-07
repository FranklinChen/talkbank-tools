use super::*;
use std::collections::BTreeSet;

fn words(text: &str) -> Vec<String> {
    text.split_whitespace().map(str::to_owned).collect()
}

/// Enumerate assignments directly, without DP states, scores or skip edges.
fn oracle(
    first: &[String],
    second: &[String],
    reference: &[String],
) -> Vec<BTreeSet<MatchAddress>> {
    fn extend(
        sources: [&[String]; 2],
        reference: &[String],
        next: [usize; 2],
        r: usize,
        matches: &mut BTreeSet<MatchAddress>,
        assignments: &mut Vec<BTreeSet<MatchAddress>>,
    ) {
        assignments.push(matches.clone());
        for chain in [SpeakerChain::First, SpeakerChain::Second] {
            for word in next[chain.index()]..sources[chain.index()].len() {
                for reference_index in r..reference.len() {
                    if sources[chain.index()][word] != reference[reference_index] {
                        continue;
                    }
                    let address = MatchAddress {
                        chain,
                        word,
                        reference: reference_index,
                    };
                    matches.insert(address);
                    let mut next = next;
                    next[chain.index()] = word + 1;
                    extend(
                        sources,
                        reference,
                        next,
                        reference_index + 1,
                        matches,
                        assignments,
                    );
                    matches.remove(&address);
                }
            }
        }
    }
    let mut assignments = Vec::new();
    extend(
        [first, second],
        reference,
        [0, 0],
        0,
        &mut BTreeSet::new(),
        &mut assignments,
    );
    let optimum = assignments.iter().map(BTreeSet::len).max().unwrap();
    assignments
        .into_iter()
        .filter(|assignment| assignment.len() == optimum)
        .collect()
}

fn binary_sequences(max_len: usize) -> Vec<Vec<String>> {
    let mut sequences = vec![Vec::new()];
    for length in 1..=max_len {
        for bits in 0..(1usize << length) {
            sequences.push(
                (0..length)
                    .map(|i| if bits & (1 << i) == 0 { "a" } else { "b" }.to_owned())
                    .collect(),
            );
        }
    }
    sequences
}

#[test]
fn product_dag_matches_independent_exhaustive_assignments() {
    let sources = binary_sequences(2);
    let references = binary_sequences(3);
    for first in &sources {
        for second in &sources {
            for reference in &references {
                let expected = oracle(first, second, reference);
                let analysis =
                    TwoSpeakerAlignment::observe(first, second, reference, MatchMode::Exact)
                        .unwrap();
                let selected: BTreeSet<_> =
                    analysis.selected().map(|matched| matched.address).collect();
                assert!(
                    expected.contains(&selected),
                    "{first:?} / {second:?} / {reference:?}"
                );
                assert_eq!(analysis.matched_words(), selected.len());
                let reference_population: BTreeSet<_> =
                    selected.iter().map(|matched| matched.reference).collect();
                assert_eq!(
                    reference_population.len(),
                    selected.len(),
                    "reference assignments are exclusive"
                );
                for chain in [SpeakerChain::First, SpeakerChain::Second] {
                    for (word, evidence) in analysis.words(chain).enumerate() {
                        let candidates: BTreeSet<_> = expected
                            .iter()
                            .flat_map(|assignment| assignment.iter())
                            .filter(|address| address.chain == chain && address.word == word)
                            .copied()
                            .collect();
                        assert_eq!(
                            evidence
                                .candidates()
                                .map(|matched| matched.address)
                                .collect::<BTreeSet<_>>(),
                            candidates
                        );
                        let missing = expected.iter().any(|assignment| {
                            !assignment
                                .iter()
                                .any(|address| address.chain == chain && address.word == word)
                        });
                        assert_eq!(evidence.can_be_missing(), missing);
                        let common = candidates.iter().copied().find(|address| {
                            expected
                                .iter()
                                .all(|assignment| assignment.contains(address))
                        });
                        assert_eq!(
                            evidence.common().map(|matched| matched.matched().address),
                            common
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn unique_cross_speaker_insertion_is_recovered_without_changing_either_chain_order() {
    let first = words("one two three");
    let second = words("yes now");
    let reference = words("one yes two now three");
    let analysis =
        TwoSpeakerAlignment::observe(&first, &second, &reference, MatchMode::Exact).unwrap();
    assert_eq!(analysis.matched_words(), 5);
    assert_eq!(
        analysis
            .words(SpeakerChain::First)
            .map(|word| word.common().unwrap().matched().reference_index())
            .collect::<Vec<_>>(),
        [0, 2, 4]
    );
    assert_eq!(
        analysis
            .words(SpeakerChain::Second)
            .map(|word| word.common().unwrap().matched().reference_index())
            .collect::<Vec<_>>(),
        [1, 3]
    );
    let reversed = words("three two one");
    let analysis = TwoSpeakerAlignment::observe(&first, &[], &reversed, MatchMode::Exact).unwrap();
    assert_eq!(
        analysis.matched_words(),
        1,
        "same-speaker order must not be relaxed"
    );
}

#[test]
fn one_reference_word_cannot_certify_two_speakers() {
    let source = words("yes");
    let reference = words("yes");
    let analysis =
        TwoSpeakerAlignment::observe(&source, &source, &reference, MatchMode::Exact).unwrap();
    assert_eq!(analysis.matched_words(), 1);
    for chain in [SpeakerChain::First, SpeakerChain::Second] {
        let word = analysis.words(chain).next().unwrap();
        assert!(word.can_be_missing());
        assert_eq!(word.candidates().count(), 1);
        assert!(
            word.common().is_none(),
            "one possible location is not a mandatory match"
        );
    }
}

#[test]
fn unmarked_repetition_retains_competing_assignments_instead_of_certifying_late_words() {
    let first = words("who are paving the way for all of you");
    let second = words("all of");
    let reference = words("who are paving the way for all all of of you all of");
    let analysis =
        TwoSpeakerAlignment::observe(&first, &second, &reference, MatchMode::CaseInsensitive)
            .unwrap();
    assert_eq!(analysis.matched_words(), 11);
    let evidence = analysis
        .words(SpeakerChain::Second)
        .map(|word| {
            format!(
                "missing={} common={:?} possible={:?}",
                word.can_be_missing(),
                word.common()
                    .map(|matched| matched.matched().reference_index()),
                word.candidates()
                    .map(|matched| matched.reference_index())
                    .collect::<Vec<_>>()
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    insta::assert_snapshot!(evidence, @"
    missing=false common=None possible=[6, 7, 11]
    missing=false common=None possible=[8, 9, 12]
    ");
    assert!(
        analysis
            .words(SpeakerChain::First)
            .take(6)
            .all(|word| word.common().is_some())
    );
}

#[test]
fn relation_preserves_ascii_and_fuzzy_semantics() {
    let first = words("A gonna");
    let second = words("à");
    let reference = words("a À gona à");
    for (mode, count) in [
        (MatchMode::Exact, 1),
        (MatchMode::CaseInsensitive, 2),
        (MatchMode::Fuzzy { threshold: 0.85 }, 3),
    ] {
        let analysis = TwoSpeakerAlignment::observe(&first, &second, &reference, mode).unwrap();
        assert_eq!(analysis.matched_words(), count);
        for matched in analysis.selected() {
            assert_eq!(
                matched.source_text(),
                [first.as_slice(), second.as_slice()][matched.chain().index()]
                    [matched.word_index()]
            );
            assert_eq!(
                matched.reference_text(),
                reference[matched.reference_index()]
            );
        }
    }
}

#[test]
fn refusal_is_not_empty_or_ambiguous_evidence() {
    let source = words("a a");
    let reference = words("a a a");
    assert!(matches!(
        TwoSpeakerAlignment::with_budget(&source, &source, &reference, MatchMode::Exact, 35),
        Err(InterleavingRefusal::CellBudgetExceeded)
    ));
    assert!(
        TwoSpeakerAlignment::with_budget(&source, &source, &reference, MatchMode::Exact, 36)
            .is_ok()
    );
    assert!(AdmittedShape::new(usize::MAX, 0, 0, usize::MAX).is_none());
    let empty = TwoSpeakerAlignment::observe(&[], &[], &[], MatchMode::Exact).unwrap();
    assert_eq!(empty.matched_words(), 0);
    assert_eq!(empty.selected().count(), 0);
}
