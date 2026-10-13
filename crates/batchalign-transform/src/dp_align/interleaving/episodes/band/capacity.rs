//! Source-bound literal bag upper bounds, never correspondence certificates.
//! At reference cut k, no prefix/suffix can match more than its bag intersection
//! with the whole source. Using the whole source on both sides overestimates
//! competition deliberately; it cannot remove a globally optimal assignment.

use super::*;
use crate::dp_align::LiteralMatchMode;
use crate::dp_align::packed_literal::policy;
use std::borrow::Cow;
use std::collections::HashMap;

/// Only completed observation can construct this relation-bound capacity.
/// Construction temporaries die on interruption; no partial bound escapes.
pub(super) struct ReferenceCapacity<'relation, 'source> {
    source: &'relation [&'source str],
    prefix: Vec<usize>,
    suffix: Vec<usize>,
    work: usize,
    feasible: usize,
}

impl<'relation, 'source> ReferenceCapacity<'relation, 'source> {
    pub(super) fn observe<C: WorkControl + ?Sized>(
        witness: &CanonicalWitness<'relation, 'source>,
        control: &C,
    ) -> Result<Option<Self>, C::Error> {
        let relation = witness.relation();
        let Relation::Literal {
            source,
            reference,
            mode,
        } = relation
        else {
            return Ok(None);
        };
        let Some(mode) = policy(*mode) else {
            return Ok(None);
        };
        // Reserve both retained vectors and a conservative map-entry/header
        // allowance before constructing the temporary counts. Like the score
        // budget, this is machine-word payload, not allocator/RSS accounting.
        let cells = reference.len().saturating_add(1).saturating_mul(2);
        let temporary = source.len().saturating_mul(32).saturating_add(cells);
        let work = source
            .len()
            .saturating_add(reference.len())
            .saturating_mul(6)
            .saturating_add(2);
        if temporary > SCORE_CELLS || work.saturating_add(witness.work()) > WORK_CELLS {
            return Ok(None);
        }
        control.check()?;
        let key = |word: &'source str| match mode {
            LiteralMatchMode::Exact => Cow::Borrowed(word),
            LiteralMatchMode::AsciiInsensitive => Cow::Owned(word.to_ascii_lowercase()),
        };
        let mut counts: HashMap<Cow<'source, str>, (usize, usize)> = HashMap::new();
        for (index, word) in source.iter().enumerate() {
            if index % 1024 == 0 {
                control.check()?;
            }
            counts.entry(key(word)).or_default().0 += 1;
        }
        let mut prefix = Vec::with_capacity(reference.len() + 1);
        prefix.push(0);
        let mut matched = 0;
        for (index, word) in reference.iter().enumerate() {
            if index % 1024 == 0 {
                control.check()?;
            }
            if let Some((available, seen)) = counts.get_mut(key(word).as_ref()) {
                matched += usize::from(*seen < *available);
                *seen += 1;
            }
            prefix.push(matched);
        }
        for (index, (_, seen)) in counts.values_mut().enumerate() {
            if index % 1024 == 0 {
                control.check()?;
            }
            *seen = 0;
        }
        let mut suffix = vec![0; reference.len() + 1];
        matched = 0;
        for (index, word) in reference.iter().enumerate().rev() {
            if index % 1024 == 0 {
                control.check()?;
            }
            if let Some((available, seen)) = counts.get_mut(key(word).as_ref()) {
                matched += usize::from(*seen < *available);
                *seen += 1;
            }
            suffix[index] = matched;
        }
        Ok(Some(Self {
            source,
            prefix,
            suffix,
            work,
            feasible: witness.score(),
        }))
    }

    pub(super) fn work(&self) -> usize {
        self.work
    }
    pub(super) fn cells(&self) -> usize {
        self.prefix.len() + self.suffix.len()
    }

    pub(super) fn depths(&self, k: usize) -> (usize, usize) {
        // min(d,C)+min(m-d,D) >= L requires both these bounds. No
        // word order, correspondence choice or acoustic claim is introduced.
        depths(
            self.source.len(),
            self.prefix[k],
            self.suffix[k],
            self.feasible,
        )
    }
}

fn depths(m: usize, prefix: usize, suffix: usize, feasible: usize) -> (usize, usize) {
    (
        feasible.saturating_sub(suffix),
        m.min(m - feasible + prefix),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(words: &[&str]) -> Vec<String> {
        words.iter().map(|word| (*word).to_owned()).collect()
    }

    #[test]
    fn capacities_count_multiplicity_not_presence_and_fold_only_ascii() {
        let source = ["a", "a", "B", "é"];
        let reference = strings(&["A", "a", "a", "b", "É", "é"]);
        let observe = |mode| {
            let relation = Relation::new(source.to_vec(), &reference, mode);
            let capacity =
                ReferenceCapacity::observe(&CanonicalWitness::zero(&relation), &Uninterrupted)
                    .unwrap()
                    .unwrap();
            (capacity.prefix, capacity.suffix)
        };
        insta::assert_snapshot!(format!("{:?}", observe(MatchMode::Exact)),
            @"([0, 0, 1, 2, 2, 2, 3], [3, 3, 2, 1, 1, 1, 0])");
        assert_eq!(
            observe(MatchMode::CaseInsensitive),
            (vec![0, 1, 2, 2, 3, 3, 4], vec![4, 4, 3, 2, 1, 1, 0])
        );
    }

    #[test]
    fn fuzzy_and_unaffordable_construction_do_not_supply_literal_bounds() {
        let reference = strings(&["a"]);
        let fuzzy = Relation::new(vec!["a"], &reference, MatchMode::Fuzzy { threshold: 0.8 });
        assert!(
            ReferenceCapacity::observe(&CanonicalWitness::zero(&fuzzy), &Uninterrupted)
                .unwrap()
                .is_none()
        );
        let huge = Relation::new(
            vec!["a"; SCORE_CELLS / 32 + 1],
            &reference,
            MatchMode::Exact,
        );
        assert!(
            ReferenceCapacity::observe(&CanonicalWitness::zero(&huge), &Uninterrupted)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn interruption_during_prefix_observation_returns_no_bound() {
        use std::cell::Cell;
        struct Cancel(Cell<usize>);
        impl WorkControl for Cancel {
            type Error = &'static str;
            fn check(&self) -> Result<(), Self::Error> {
                self.0.set(self.0.get() + 1);
                if self.0.get() == 4 {
                    Err("cancelled")
                } else {
                    Ok(())
                }
            }
        }
        let reference = vec!["a".to_owned(); 2048];
        let relation = Relation::new(vec!["a"], &reference, MatchMode::Exact);
        assert!(matches!(
            ReferenceCapacity::observe(&CanonicalWitness::zero(&relation), &Cancel(Cell::new(0))),
            Err("cancelled")
        ));
    }

    #[test]
    fn every_small_bag_partition_preserves_every_feasible_matching_count() {
        // For any path and reference cut, p prefix + q suffix matches are
        // bounded by source depth and these capacities. Enumerate all possible
        // such counts, including repetition, empty ends and zero score.
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
                        let relation = Relation::new(source.clone(), &reference, MatchMode::Exact);
                        let capacity = ReferenceCapacity::observe(
                            &CanonicalWitness::zero(&relation),
                            &Uninterrupted,
                        )
                        .unwrap()
                        .unwrap();
                        for k in 0..=n {
                            let bag = |slice: &[String]| {
                                ["a", "b"]
                                    .iter()
                                    .map(|word| {
                                        source.iter().filter(|s| *s == word).count().min(
                                            slice.iter().filter(|r| r.as_str() == *word).count(),
                                        )
                                    })
                                    .sum::<usize>()
                            };
                            assert_eq!(capacity.prefix[k], bag(&reference[..k]));
                            assert_eq!(capacity.suffix[k], bag(&reference[k..]));
                            for d in 0..=m {
                                let upper =
                                    d.min(capacity.prefix[k]) + (m - d).min(capacity.suffix[k]);
                                for feasible in 0..=upper {
                                    let (low, high) =
                                        depths(m, capacity.prefix[k], capacity.suffix[k], feasible);
                                    assert!(low <= d && d <= high);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
