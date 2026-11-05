//! A producer-bound feasible score. It admits pruning, never addressed matches.

use super::*;
use crate::dp_align::LiteralMatchMode;
use crate::dp_align::packed_literal::{BIT_WORDS, LiteralReference, ReferenceOrder};

mod preparation;
use preparation::PreparedWitness;

struct InteriorScore(usize);

struct LiteralInterior<'relation, 'source> {
    source: &'relation [&'source str],
    reference: &'relation [String],
    mode: LiteralMatchMode,
}

/// A feasible canonical score, never an addressed correspondence certificate.
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
        control: &C,
    ) -> Result<Result<Self, LocalInterleavingRefusal>, C::Error> {
        match PreparedWitness::observe(relation, control)? {
            Ok(prepared) => prepared.finish(control).map(Ok),
            Err(refusal) => Ok(Err(refusal)),
        }
    }
}

impl LiteralInterior<'_, '_> {
    // One sparse-mask/row stream per source word, plus normalization/scans.
    // This is declared algorithmic work, not latency or byte memory.
    fn work(&self) -> usize {
        self.source
            .len()
            .saturating_mul(self.reference.len().div_ceil(64))
            .saturating_add(self.source.len().saturating_mul(3))
            .saturating_add(self.reference.len().saturating_mul(3))
            .saturating_add(2)
    }

    fn cells(&self) -> usize {
        // Posting pairs, map entries and two packed rows.
        self.reference.len().saturating_mul(6).saturating_add(
            self.reference
                .len()
                .div_ceil(64)
                .saturating_mul(2 * BIT_WORDS),
        )
    }

    /// Allison-Dix exact LCS row differences with sparse equality masks.
    /// https://www.cantab.net/users/mmlist/ll/Publications/1986IPL/
    /// Only the admitted witness owner executes this in production.
    fn execute<C: WorkControl + ?Sized>(self, control: &C) -> Result<InteriorScore, C::Error> {
        let reference =
            LiteralReference::observe(self.reference, self.mode, ReferenceOrder::Forward, control)?;
        let mut row = reference.start();
        for word in self.source {
            row = row.extend(word, control)?;
        }
        // Padding bits cannot enter a state: masks contain real reference bits.
        Ok(InteriorScore(row.score()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_witness_admits_only_the_unmatched_interior() {
        let source = vec!["a"; 35_000];
        let mut reference = vec!["a".to_owned(); 40_000];
        reference[0] = "b".to_owned();
        reference[39_999] = "b".to_owned();
        let relation = Relation::new(source, &reference, MatchMode::Exact);
        let witness = uninterrupted(CanonicalWitness::observe(&relation, &Uninterrupted)).unwrap();
        assert_eq!(witness.score(), 35_000);
        assert!(witness.work() <= WORK_CELLS);

        // The matching edges are already a completed part of the witness.
        // Charging/scoring them again would refuse this affordable interior.
        let mut source = vec!["edge"; 66_000];
        let mut reference = vec!["edge".to_owned(); 66_000];
        source[32_000..34_000].fill("source-only");
        reference[32_000..34_000].fill("reference-only".to_owned());
        let relation = Relation::new(source, &reference, MatchMode::Exact);
        let witness = uninterrupted(CanonicalWitness::observe(&relation, &Uninterrupted))
            .expect("only the 2,000 by 2,000 unmatched interior needs scoring");
        assert_eq!(witness.score(), 64_000);
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
    fn canonical_witness_equals_scalar_oracle_across_word_boundaries_and_repetitions() {
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
                        let relation = Relation::new(source.clone(), &reference, mode);
                        let witness =
                            uninterrupted(CanonicalWitness::observe(&relation, &Uninterrupted))
                                .unwrap();
                        assert_eq!(
                            witness.score(),
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
        assert!(matches!(
            CanonicalWitness::observe(
                &Relation::new(vec!["a"], &["a".to_owned()], MatchMode::Exact),
                &Cancel,
            ),
            Err("cancelled")
        ));
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
                        let InteriorScore(score) = LiteralInterior {
                            source: &source,
                            reference: &reference,
                            mode: LiteralMatchMode::Exact,
                        }
                        .execute(&Uninterrupted)
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
            let InteriorScore(score) = LiteralInterior {
                source: &source,
                reference: &reference,
                mode: LiteralMatchMode::Exact,
            }
            .execute(&Uninterrupted)
            .unwrap();
            assert_eq!(score, m.min(257));
        }
    }

    #[test]
    fn canonical_witness_admits_work_and_memory_before_row_allocation() {
        // Packed masks do not fit, but the scalar row and its work do. A
        // representation's refusal is not refusal of the canonical problem.
        let reference = vec!["reference".to_owned(); 350_000];
        let relation = Relation::new(vec!["source"; 4], &reference, MatchMode::Exact);
        let witness = uninterrupted(CanonicalWitness::observe(&relation, &Uninterrupted))
            .expect("the admitted scalar alternative must remain available");
        assert_eq!(witness.score(), 0);

        let source = vec!["source"; 1_000_000];
        let reference = vec!["reference".to_owned(); 8192];
        assert!(matches!(
            uninterrupted(CanonicalWitness::observe(
                &Relation::new(source, &reference, MatchMode::Exact),
                &Uninterrupted,
            )),
            Err(LocalInterleavingRefusal::Budget(
                LocalInterleavingBudget::WorkBudgetExceeded {
                    phase: LocalInterleavingWorkPhase::FeasibleWitness,
                    ..
                }
            ))
        ));
        let source = vec!["source"; 1000];
        let reference = vec!["reference".to_owned(); 400_000];
        assert!(matches!(
            uninterrupted(CanonicalWitness::observe(
                &Relation::new(source, &reference, MatchMode::Exact),
                &Uninterrupted,
            )),
            Err(LocalInterleavingRefusal::Budget(
                LocalInterleavingBudget::MemoryBudgetExceeded
            ))
        ));
    }
}
