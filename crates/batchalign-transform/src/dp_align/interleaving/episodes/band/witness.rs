//! A producer-bound feasible score. It admits pruning, never addressed matches.

use super::*;
use crate::dp_align::LiteralMatchMode;
use crate::dp_align::packed_literal::{BIT_WORDS, LiteralReference, ReferenceOrder, policy};

pub(super) struct CanonicalWitness<'relation, 'source> {
    relation: &'relation Relation<'source>,
    score: usize,
    work: usize,
}

impl<'relation, 'source> CanonicalWitness<'relation, 'source> {
    pub(super) fn zero(relation: &'relation Relation<'source>) -> Self {
        Self {
            relation,
            score: 0,
            work: 0,
        }
    }

    pub(super) fn relation(&self) -> &'relation Relation<'source> {
        self.relation
    }
    pub(super) fn score(&self) -> usize {
        self.score
    }
    pub(super) fn work(&self) -> usize {
        self.work
    }

    pub(super) fn observe<C: WorkControl + ?Sized>(
        relation: &'relation Relation<'source>,
        m: usize,
        n: usize,
        control: &C,
    ) -> Result<Result<Self, LocalInterleavingRefusal>, C::Error> {
        if m.max(n) > WORK_CELLS {
            return Ok(Err(LocalInterleavingRefusal::WorkBudgetExceeded {
                phase: LocalInterleavingWorkPhase::FeasibleWitness,
                required_at_least: m.max(n),
            }));
        }
        let mut prefix = 0;
        while prefix < m.min(n) && relation.matches(prefix, prefix) {
            if prefix % 1024 == 0 {
                control.check()?;
            }
            prefix += 1;
        }
        let mut suffix = 0;
        while suffix < m.min(n) - prefix && relation.matches(m - 1 - suffix, n - 1 - suffix) {
            if suffix % 1024 == 0 {
                control.check()?;
            }
            suffix += 1;
        }
        // Retain the zero-/small-work prefix/suffix path and general fuzzy
        // relation. A literal word equality supports bit-parallel exact LCS.
        let quadratic = (m - prefix - suffix).saturating_mul(n - prefix - suffix);
        let result = if let Relation::Literal {
            source,
            reference,
            mode,
        } = relation
            && quadratic > 1_000_000
            && let Some(mode) = policy(*mode)
        {
            literal(source, reference, mode, control)?
        } else {
            super::witness(relation, m, n, control)?
        };
        Ok(result.and_then(|(score, work)| {
            let work = work.saturating_add(prefix + suffix);
            if work > WORK_CELLS {
                Err(LocalInterleavingRefusal::WorkBudgetExceeded {
                    phase: LocalInterleavingWorkPhase::FeasibleWitness,
                    required_at_least: work,
                })
            } else {
                Ok(Self {
                    relation,
                    score,
                    work,
                })
            }
        }))
    }
}

/// Allison-Dix exact LCS row differences packed into machine words.
/// Sparse equality masks avoid alphabet-size times transcript-size storage.
/// https://www.cantab.net/users/mmlist/ll/Publications/1986IPL/
fn literal<C: WorkControl + ?Sized>(
    source: &[&str],
    reference: &[String],
    mode: LiteralMatchMode,
    control: &C,
) -> Result<Result<(usize, usize), LocalInterleavingRefusal>, C::Error> {
    let blocks = reference.len().div_ceil(64);
    // One sparse-mask/row stream per source word. Include preprocessing,
    // normalization and scans; there is no dense mask clear/populate pass.
    let work = source
        .len()
        .saturating_mul(blocks)
        .saturating_add(source.len().saturating_mul(3))
        .saturating_add(reference.len().saturating_mul(3))
        .saturating_add(2);
    if work > WORK_CELLS {
        return Ok(Err(LocalInterleavingRefusal::WorkBudgetExceeded {
            phase: LocalInterleavingWorkPhase::FeasibleWitness,
            required_at_least: work,
        }));
    }
    // Posting pairs, map entries and two packed rows, before any allocation.
    if reference
        .len()
        .saturating_mul(6)
        .saturating_add(blocks.saturating_mul(2 * BIT_WORDS))
        > SCORE_CELLS
    {
        return Ok(Err(LocalInterleavingRefusal::MemoryBudgetExceeded));
    }
    let reference = LiteralReference::observe(reference, mode, ReferenceOrder::Forward, control)?;
    let mut row = reference.start();
    for word in source {
        row = row.extend(word, control)?;
    }
    // Padding bits cannot enter a state: x contains only real reference bits.
    Ok(Ok((row.score(), work)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_streaming_witness_admits_long_literal_rows_under_the_unchanged_limit() {
        let source = vec!["a"; 35_000];
        let mut reference = vec!["a".to_owned(); 40_000];
        reference[0] = "b".to_owned();
        reference[39_999] = "b".to_owned();
        let relation = Relation::new(source, &reference, MatchMode::Exact);
        let witness = uninterrupted(CanonicalWitness::observe(
            &relation,
            35_000,
            40_000,
            &Uninterrupted,
        ))
        .unwrap();
        assert_eq!(witness.score(), 35_000);
        assert!(witness.work() <= WORK_CELLS);
    }

    fn oracle(source: &[&str], reference: &[String], mode: MatchMode) -> usize {
        let mut row = vec![0usize; reference.len() + 1];
        for source in source {
            let mut diagonal = 0;
            for (j, reference) in reference.iter().enumerate() {
                let old = row[j + 1];
                let equal = match mode {
                    MatchMode::Exact => *source == reference,
                    _ => source.eq_ignore_ascii_case(reference),
                };
                row[j + 1] = old.max(row[j]).max(diagonal + usize::from(equal));
                diagonal = old;
            }
        }
        row[reference.len()]
    }

    #[test]
    fn packed_witness_equals_scalar_oracle_across_word_boundaries_and_repetitions() {
        let alphabet = ["a", "A", "b", "B", "é", "É", "other"];
        for m in [0, 1, 63, 64, 65, 127, 128, 129, 257] {
            for n in [0, 1, 63, 64, 65, 127, 128, 129, 257] {
                for seed in 0..8 {
                    let source: Vec<_> = (0..m)
                        .map(|i| alphabet[(i * 13 + seed) % alphabet.len()])
                        .collect();
                    let reference: Vec<_> = (0..n)
                        .map(|i| alphabet[(i * 3 + i / 11 + seed * 2) % alphabet.len()].to_owned())
                        .collect();
                    for mode in [MatchMode::Exact, MatchMode::CaseInsensitive] {
                        let (score, _) = uninterrupted(literal(
                            &source,
                            &reference,
                            policy(mode).unwrap(),
                            &Uninterrupted,
                        ))
                        .unwrap();
                        assert_eq!(
                            score,
                            oracle(&source, &reference, mode),
                            "m={m} n={n} seed={seed} mode={mode:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn packed_witness_cancellation_is_not_a_budget_or_correspondence_result() {
        struct Cancel;
        impl WorkControl for Cancel {
            type Error = &'static str;
            fn check(&self) -> Result<(), Self::Error> {
                Err("cancelled")
            }
        }
        assert_eq!(
            literal(&["a"], &["a".to_owned()], LiteralMatchMode::Exact, &Cancel),
            Err("cancelled")
        );
    }

    #[test]
    fn packed_witness_equals_oracle_for_every_small_binary_pair() {
        for m in 0..=5 {
            for n in 0..=6 {
                for a in 0..1usize << m {
                    let source: Vec<_> = (0..m)
                        .map(|i| if a & (1 << i) == 0 { "a" } else { "b" })
                        .collect();
                    for b in 0..1usize << n {
                        let reference: Vec<_> = (0..n)
                            .map(|i| if b & (1 << i) == 0 { "a" } else { "b" }.to_owned())
                            .collect();
                        let (score, _) = uninterrupted(literal(
                            &source,
                            &reference,
                            LiteralMatchMode::Exact,
                            &Uninterrupted,
                        ))
                        .unwrap();
                        assert_eq!(score, oracle(&source, &reference, MatchMode::Exact));
                    }
                }
            }
        }
    }

    #[test]
    fn repeated_matches_propagate_carry_and_borrow_across_all_blocks() {
        for m in [63, 64, 65, 128, 257, 300] {
            let source = vec!["same"; m];
            let reference = vec!["same".to_owned(); 257];
            let (score, _) = uninterrupted(literal(
                &source,
                &reference,
                LiteralMatchMode::Exact,
                &Uninterrupted,
            ))
            .unwrap();
            assert_eq!(score, m.min(257));
        }
    }

    #[test]
    fn packed_witness_admits_work_and_memory_before_mask_allocation() {
        let source = vec!["a"; 1_000_000];
        let reference = vec!["a".to_owned(); 8192];
        assert!(matches!(
            uninterrupted(literal(
                &source,
                &reference,
                LiteralMatchMode::Exact,
                &Uninterrupted
            )),
            Err(LocalInterleavingRefusal::WorkBudgetExceeded {
                phase: LocalInterleavingWorkPhase::FeasibleWitness,
                ..
            })
        ));
        let source = vec!["a"; 1000];
        let reference = vec!["a".to_owned(); 400_000];
        assert!(matches!(
            uninterrupted(literal(
                &source,
                &reference,
                LiteralMatchMode::Exact,
                &Uninterrupted
            )),
            Err(LocalInterleavingRefusal::MemoryBudgetExceeded)
        ));
    }
}
