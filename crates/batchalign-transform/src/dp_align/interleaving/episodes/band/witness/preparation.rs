//! Edge matching is paid once; only its owned interior can reach a scorer.

use super::*;

pub(super) enum PreparedWitness<'relation, 'source> {
    Complete(CanonicalWitness<'relation, 'source>),
    Interior(AdmittedInterior<'relation, 'source>),
}

struct WitnessInterior<'relation, 'source> {
    relation: &'relation Relation<'source>,
    source: Range<usize>,
    reference: Range<usize>,
    matched_edges: usize,
}

impl<'relation, 'source> PreparedWitness<'relation, 'source> {
    pub(super) fn observe<C: WorkControl + ?Sized>(
        relation: &'relation Relation<'source>,
        control: &C,
    ) -> Result<Result<Self, LocalInterleavingRefusal>, C::Error> {
        control.check()?;
        let m = match relation {
            Relation::Literal { source, .. } => source.len(),
            Relation::Fuzzy { source, .. } => source.len(),
        };
        let n = match relation {
            Relation::Literal { reference, .. } => reference.len(),
            Relation::Fuzzy { reference, .. } => reference.len(),
        };
        if m.max(n) > WORK_CELLS {
            return Ok(Err(refusal(m.max(n))));
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
        let matched_edges = prefix + suffix;
        // If either interior is empty, the matched edges are the whole score.
        // This variant has no scorer or row-allocation capability.
        Ok(Ok(if matched_edges == m.min(n) {
            Self::Complete(CanonicalWitness {
                relation,
                score: matched_edges,
                work: matched_edges,
            })
        } else {
            let interior = WitnessInterior {
                relation,
                source: prefix..m - suffix,
                reference: prefix..n - suffix,
                matched_edges,
            };
            return Ok(interior.admit().map(Self::Interior));
        }))
    }

    pub(super) fn finish<C: WorkControl + ?Sized>(
        self,
        control: &C,
    ) -> Result<CanonicalWitness<'relation, 'source>, C::Error> {
        match self {
            Self::Complete(witness) => Ok(witness),
            Self::Interior(interior) => interior.finish(control),
        }
    }
}

/// Resource checks are on the only edge into execution, before row allocation.
/// The scorer cannot introduce a late budget refusal or substitute its inputs.
pub(super) struct AdmittedInterior<'relation, 'source> {
    relation: &'relation Relation<'source>,
    matched_edges: usize,
    admitted: AdmittedKernel<'relation, 'source>,
}

struct AdmittedKernel<'relation, 'source> {
    work: usize,
    kernel: InteriorKernel<'relation, 'source>,
}

enum InteriorKernel<'relation, 'source> {
    Scalar(ScalarInterior<'relation, 'source>),
    Literal(LiteralInterior<'relation, 'source>),
}

struct ScalarInterior<'relation, 'source> {
    relation: &'relation Relation<'source>,
    source: Range<usize>,
    reference: Range<usize>,
}

struct KernelProjection<'relation, 'source> {
    kernel: InteriorKernel<'relation, 'source>,
    work: usize,
    cells: usize,
}

impl<'relation, 'source> WitnessInterior<'relation, 'source> {
    fn admit(self) -> Result<AdmittedInterior<'relation, 'source>, LocalInterleavingRefusal> {
        let scalar = ScalarInterior {
            relation: self.relation,
            source: self.source,
            reference: self.reference,
        };
        let admitted = match self.relation {
            Relation::Literal {
                source,
                reference,
                mode,
            } => {
                let literal = LiteralInterior {
                    source: &source[scalar.source.clone()],
                    reference: &reference[scalar.reference.clone()],
                    mode: *mode,
                };
                let literal = KernelProjection {
                    work: literal.work(),
                    cells: literal.cells(),
                    kernel: InteriorKernel::Literal(literal),
                };
                KernelProjection::scalar(scalar).admit_alternative(literal, self.matched_edges)?
            }
            Relation::Fuzzy { .. } => KernelProjection::scalar(scalar).admit(self.matched_edges)?,
        };
        Ok(AdmittedInterior {
            relation: self.relation,
            matched_edges: self.matched_edges,
            admitted,
        })
    }
}

impl<'relation, 'source> KernelProjection<'relation, 'source> {
    fn scalar(scalar: ScalarInterior<'relation, 'source>) -> Self {
        Self {
            work: scalar
                .source
                .len()
                .saturating_mul(scalar.reference.len())
                .saturating_add(2),
            cells: scalar.reference.len().saturating_add(1),
            kernel: InteriorKernel::Scalar(scalar),
        }
    }

    fn admit_alternative(
        self,
        alternative: Self,
        matched_edges: usize,
    ) -> Result<AdmittedKernel<'relation, 'source>, LocalInterleavingRefusal> {
        let [preferred, alternate] = if self.work <= alternative.work {
            [self, alternative]
        } else {
            [alternative, self]
        };
        // Neither projection has allocated or executed a row. A representation
        // refusal retains the other candidate; execution can still happen once.
        // If both refuse, retain the preferred representation's diagnostic.
        preferred.admit(matched_edges).or_else(|preferred_refusal| {
            alternate
                .admit(matched_edges)
                .map_err(|_| preferred_refusal)
        })
    }

    fn admit(
        self,
        matched_edges: usize,
    ) -> Result<AdmittedKernel<'relation, 'source>, LocalInterleavingRefusal> {
        let work = self.work.saturating_add(matched_edges);
        if work > WORK_CELLS {
            return Err(refusal(work));
        }
        if self.cells > SCORE_CELLS {
            return Err(LocalInterleavingRefusal::Budget(
                LocalInterleavingBudget::MemoryBudgetExceeded,
            ));
        }
        Ok(AdmittedKernel {
            work,
            kernel: self.kernel,
        })
    }
}

impl<'relation, 'source> AdmittedInterior<'relation, 'source> {
    fn finish<C: WorkControl + ?Sized>(
        self,
        control: &C,
    ) -> Result<CanonicalWitness<'relation, 'source>, C::Error> {
        let InteriorScore(score) = match self.admitted.kernel {
            InteriorKernel::Scalar(scalar) => scalar.execute(control)?,
            InteriorKernel::Literal(literal) => literal.execute(control)?,
        };
        Ok(CanonicalWitness {
            relation: self.relation,
            score: self.matched_edges + score,
            work: self.admitted.work,
        })
    }
}

impl ScalarInterior<'_, '_> {
    fn execute<C: WorkControl + ?Sized>(self, control: &C) -> Result<InteriorScore, C::Error> {
        let b = self.reference.len();
        let mut row = vec![0usize; b + 1];
        for i in self.source {
            control.check()?;
            let mut diagonal = 0;
            for j in 0..b {
                if j % 1024 == 0 {
                    control.check()?;
                }
                let old = row[j + 1];
                row[j + 1] = old.max(row[j]);
                if self.relation.matches(i, self.reference.start + j) {
                    row[j + 1] = row[j + 1].max(diagonal + 1);
                }
                diagonal = old;
            }
        }
        Ok(InteriorScore(row[b]))
    }
}

fn refusal(required_at_least: usize) -> LocalInterleavingRefusal {
    LocalInterleavingRefusal::Budget(LocalInterleavingBudget::WorkBudgetExceeded {
        phase: LocalInterleavingWorkPhase::FeasibleWitness,
        required_at_least,
    })
}
