use super::*;
use std::collections::BTreeSet;

fn words(text: &str) -> Vec<String> {
    text.split_whitespace().map(str::to_owned).collect()
}

/// Independent legal-order and matching-assignment enumeration, not graph DP.
fn oracle(turns: &[SpeakerTurn<'_>], reference: &[String]) -> Vec<BTreeSet<(usize, usize, usize)>> {
    fn shuffle(a: &[Address], b: &[Address]) -> Vec<Vec<Address>> {
        if a.is_empty() {
            return vec![b.to_vec()];
        }
        if b.is_empty() {
            return vec![a.to_vec()];
        }
        let mut orders = Vec::new();
        for mut tail in shuffle(&a[1..], b) {
            tail.insert(0, a[0]);
            orders.push(tail);
        }
        for mut tail in shuffle(a, &b[1..]) {
            tail.insert(0, b[0]);
            orders.push(tail);
        }
        orders
    }
    fn orders(turns: &[SpeakerTurn<'_>], ordinal: usize) -> Vec<Vec<Address>> {
        if ordinal == turns.len() {
            return vec![Vec::new()];
        }
        let census = |u: usize| {
            (0..turns[u].words.len())
                .map(|word| Address { utterance: u, word })
                .collect::<Vec<_>>()
        };
        let first = census(ordinal);
        let mut result = orders(turns, ordinal + 1)
            .into_iter()
            .map(|tail| first.iter().copied().chain(tail).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        if ordinal + 1 < turns.len() && turns[ordinal].speaker != turns[ordinal + 1].speaker {
            let mut second = Vec::new();
            let speaker = turns[ordinal + 1].speaker;
            for last in ordinal + 1..turns.len() {
                if turns[last].speaker != speaker {
                    break;
                }
                if last > ordinal + 1
                    && turns[last].continuation != Continuation::OverlappingSpeakerRun
                {
                    break;
                }
                second.extend(census(last));
                for head in shuffle(&first, &second) {
                    for tail in orders(turns, last + 1) {
                        result.push(head.iter().copied().chain(tail).collect());
                    }
                }
            }
        }
        result
    }
    fn assignments(
        turns: &[SpeakerTurn<'_>],
        reference: &[String],
        order: &[Address],
        p: usize,
        r: usize,
        path: &mut BTreeSet<(usize, usize, usize)>,
        all: &mut Vec<BTreeSet<(usize, usize, usize)>>,
    ) {
        all.push(path.clone());
        for i in p..order.len() {
            let word = order[i];
            for j in r..reference.len() {
                if turns[word.utterance].words[word.word] == reference[j] {
                    path.insert((word.utterance, word.word, j));
                    assignments(turns, reference, order, i + 1, j + 1, path, all);
                    path.remove(&(word.utterance, word.word, j));
                }
            }
        }
    }
    let mut all = Vec::new();
    for order in orders(turns, 0) {
        assignments(
            turns,
            reference,
            &order,
            0,
            0,
            &mut BTreeSet::new(),
            &mut all,
        );
    }
    let maximum = all.iter().map(BTreeSet::len).max().unwrap();
    all.into_iter().filter(|p| p.len() == maximum).collect()
}

fn check(turns: &[SpeakerTurn<'_>], reference: &[String]) {
    let expected = oracle(turns, reference);
    let analysis = LocalInterleaving::observe(turns, reference, MatchMode::Exact).unwrap();
    let address = |m: LocalMatch<'_>| (m.utterance_index(), m.word_index(), m.reference_index());
    let selected = analysis.selected().map(address).collect::<BTreeSet<_>>();
    assert!(expected.contains(&selected));
    assert_eq!(selected.len(), analysis.matched_words());
    for (a, word) in analysis.addresses.iter().zip(analysis.words()) {
        let candidates = expected
            .iter()
            .flat_map(|p| p.iter())
            .filter(|(u, w, _)| *u == a.utterance && *w == a.word)
            .copied()
            .collect::<BTreeSet<_>>();
        assert_eq!(
            word.candidates().map(address).collect::<BTreeSet<_>>(),
            candidates
        );
        let missing = expected
            .iter()
            .any(|p| !p.iter().any(|(u, w, _)| *u == a.utterance && *w == a.word));
        assert_eq!(word.can_be_missing(), missing);
        let common = candidates
            .into_iter()
            .find(|c| expected.iter().all(|p| p.contains(c)));
        assert_eq!(word.common().map(|m| address(m.matched())), common);
    }
    // Independent assignment enumeration checks the new search capability,
    // including alternative optimal episode choices, not just a traceback.
    for ordinal in 0..turns.len() {
        let corridor = analysis.search_corridor(ordinal).unwrap();
        for assignment in &expected {
            let target: Vec<_> = assignment
                .iter()
                .filter(|(u, _, _)| *u == ordinal)
                .collect();
            for bound in corridor.preceding() {
                assert!(
                    target
                        .iter()
                        .all(|(_, _, r)| bound.matched().reference_index() < *r)
                );
            }
            for bound in corridor.following() {
                assert!(
                    target
                        .iter()
                        .all(|(_, _, r)| bound.matched().reference_index() > *r)
                );
            }
            for bound in corridor.preceding_required() {
                let first = bound
                    .candidates()
                    .map(|m| m.reference_index())
                    .min()
                    .unwrap();
                assert!(target.iter().all(|(_, _, r)| first < *r));
            }
            for bound in corridor.following_required() {
                let last = bound
                    .candidates()
                    .map(|m| m.reference_index())
                    .max()
                    .unwrap();
                assert!(target.iter().all(|(_, _, r)| last > *r));
            }
        }
    }
}

#[test]
fn search_corridor_is_outside_all_overlap_choices_not_adjacent_document_turns() {
    let lexical = [
        words("before"),
        words("host missing"),
        words("backchannel"),
        words("continuation"),
        words("after"),
    ];
    let turns = [
        SpeakerTurn::new("A", &lexical[0]),
        SpeakerTurn::new("A", &lexical[1]),
        SpeakerTurn::new("B", &lexical[2]),
        SpeakerTurn::overlapping_continuation("B", &lexical[3]),
        SpeakerTurn::new("B", &lexical[4]),
    ];
    let reference = words("before host backchannel continuation after");
    let analysis = LocalInterleaving::observe(&turns, &reference, MatchMode::Exact).unwrap();
    let bounds: Vec<_> = (0..turns.len())
        .map(|u| {
            let corridor = analysis.search_corridor(u).unwrap();
            (
                u,
                corridor
                    .preceding()
                    .map(|m| m.matched().source_text())
                    .collect::<Vec<_>>(),
                corridor
                    .following()
                    .map(|m| m.matched().source_text())
                    .collect::<Vec<_>>(),
            )
        })
        .collect();
    insta::assert_debug_snapshot!(bounds, @r#"
    [
        (
            0,
            [],
            [
                "host",
                "backchannel",
                "continuation",
                "after",
            ],
        ),
        (
            1,
            [
                "before",
            ],
            [
                "after",
            ],
        ),
        (
            2,
            [
                "before",
            ],
            [
                "after",
            ],
        ),
        (
            3,
            [
                "before",
            ],
            [
                "after",
            ],
        ),
        (
            4,
            [
                "continuation",
                "backchannel",
                "host",
                "before",
            ],
            [],
        ),
    ]
    "#);
}

#[test]
fn composed_episode_proofs_equal_exhaustive_legal_order_assignments() {
    let a = words("a");
    let b = words("b");
    let references = (0..=3).flat_map(|len| {
        (0..1usize << len).map(move |bits| {
            (0..len)
                .map(|i| if bits & (1 << i) == 0 { "a" } else { "b" }.to_owned())
                .collect::<Vec<_>>()
        })
    });
    for reference in references {
        for length in 0..=4 {
            for speakers in 0..1usize << length {
                for content in 0..1usize << length {
                    for overlapping in [false, true] {
                        let turns: Vec<_> = (0..length)
                            .map(|i| {
                                let speaker = if speakers & (1 << i) == 0 { "A" } else { "B" };
                                let words = if content & (1 << i) == 0 { &a } else { &b };
                                if overlapping {
                                    SpeakerTurn::overlapping_continuation(speaker, words)
                                } else {
                                    SpeakerTurn::new(speaker, words)
                                }
                            })
                            .collect();
                        check(&turns, &reference);
                    }
                }
            }
        }
    }
}

#[test]
fn empty_turns_and_multiword_pairs_preserve_the_census() {
    let a = words("a b");
    let b = words("b a");
    let empty = Vec::new();
    for reference in [words("a b a b"), words("b a b"), words("noise"), Vec::new()] {
        for turns in [
            vec![SpeakerTurn::new("A", &a), SpeakerTurn::new("B", &b)],
            vec![
                SpeakerTurn::new("A", &a),
                SpeakerTurn::new("B", &empty),
                SpeakerTurn::new("A", &b),
            ],
            vec![SpeakerTurn::new("A", &empty), SpeakerTurn::new("B", &empty)],
        ] {
            check(&turns, &reference);
        }
    }
}

#[test]
fn joint_composition_reserves_words_outside_a_pair() {
    let before = words("anchor");
    let host = words("one two three");
    let back = words("yes");
    let after = words("tail yes");
    let turns = [
        SpeakerTurn::new("A", &before),
        SpeakerTurn::new("A", &host),
        SpeakerTurn::new("B", &back),
        SpeakerTurn::new("A", &after),
    ];
    let reference = words("anchor one yes two three tail yes");
    let analysis = LocalInterleaving::observe(&turns, &reference, MatchMode::Exact).unwrap();
    assert_eq!(analysis.matched_words(), 7);
    let back = analysis.words().nth(4).unwrap();
    assert_eq!(
        back.common().unwrap().matched().reference_index(),
        2,
        "later yes is reserved by the following source turn"
    );
    check(&turns, &reference);
}

#[test]
fn block_checkpoints_recover_cross_block_common_matches() {
    let a = (0..24).map(|i| format!("word{i}")).collect::<Vec<_>>();
    let b = words("yes now");
    let turns = [SpeakerTurn::new("A", &a), SpeakerTurn::new("B", &b)];
    let mut reference = a.clone();
    reference.insert(5, "yes".into());
    reference.insert(21, "now".into());
    let analysis = LocalInterleaving::observe(&turns, &reference, MatchMode::Exact).unwrap();
    assert_eq!(analysis.matched_words(), 26);
    assert!(analysis.words().all(|word| word.common().is_some()));
    assert_eq!(analysis.selected().count(), 26);
}

#[test]
fn budget_admission_precedes_graph_construction() {
    let a = vec!["a".to_owned(); 1000];
    let reference = vec!["a".to_owned(); 100];
    let turns = [SpeakerTurn::new("A", &a), SpeakerTurn::new("B", &a)];
    assert!(matches!(
        LocalInterleaving::observe(&turns, &reference, MatchMode::Exact),
        Err(LocalInterleavingRefusal::WorkBudgetExceeded)
    ));
}

#[test]
fn a_following_speaker_run_is_one_chain_with_exclusive_repeat_assignments() {
    let host = words("one two three four five");
    let back = words("yes");
    let now = words("now");
    let turns = [
        SpeakerTurn::new("A", &host),
        SpeakerTurn::overlapping_continuation("B", &back),
        SpeakerTurn::overlapping_continuation("B", &now),
        SpeakerTurn::overlapping_continuation("B", &back),
    ];
    let reference = words("one yes two now three yes four five");
    let analysis = LocalInterleaving::observe(&turns, &reference, MatchMode::Exact).unwrap();
    assert_eq!(analysis.matched_words(), 8);
    assert_eq!(
        analysis
            .words()
            .skip(5)
            .map(|w| w.common().unwrap().matched().reference_index())
            .collect::<Vec<_>>(),
        [1, 3, 5]
    );
    check(&turns, &reference);
}

#[test]
fn ordinary_following_monologues_are_not_implicitly_one_overlap_episode() {
    let short = words("yes");
    let following = vec![words("one two"); 100];
    let turns = std::iter::once(SpeakerTurn::new("B", &short))
        .chain(following.iter().map(|words| SpeakerTurn::new("A", words)))
        .collect::<Vec<_>>();
    assert_eq!(
        episodes(&turns).count(),
        1,
        "only the adjacent turn is a local unmarked hypothesis"
    );
    let budget = Budget::admit(&turns, 201).expect("linear source size");
    assert!(
        budget.nodes < 250,
        "ordinary continuation cannot explode the source graph"
    );
}
