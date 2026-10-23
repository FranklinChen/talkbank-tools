//! Admission constructs the complete input/representation/accounting pairing.
//! No consumer can supply replacement geometry, slack or a budget policy.

use super::*;
use crate::dp_align::LiteralMatchMode;

#[derive(Clone, Copy)]
pub(super) enum RowRequest {
    Full,
    Viable,
}
struct FullRows {
    geometry: Geometry,
}
struct ViableRows {
    geometry: Geometry,
    source_slack: usize,
    reference_slack: usize,
}
enum AdmittedRows {
    Full(FullRows),
    Viable(ViableRows),
}

pub(super) struct AdmittedProblem<'g, 's> {
    refused: RefusedBand<'g, 's>,
    source: &'g [&'s str],
    reference: &'s [String],
    mode: LiteralMatchMode,
    rows: AdmittedRows,
}
impl<'g, 's> AdmittedProblem<'g, 's> {
    pub(super) fn observe<C: WorkControl + ?Sized>(
        refused: RefusedBand<'g, 's>,
        request: RowRequest,
        control: &C,
    ) -> Result<Result<Self, LocalInterleavingRefusal>, C::Error> {
        if !matches!(
            refused.refusal(),
            LocalInterleavingRefusal::Budget(LocalInterleavingBudget::WorkBudgetExceeded {
                phase: LocalInterleavingWorkPhase::BandedSearch,
                ..
            })
        ) {
            return Ok(Err(refused.refusal()));
        }
        let Relation::Literal {
            source,
            reference,
            mode,
        } = refused.relation()
        else {
            return Ok(Err(refused.refusal()));
        };
        let Some(mode) = policy(*mode) else {
            return Ok(Err(refused.refusal()));
        };
        let source_slack = source.len() - refused.feasible();
        let reference_slack = reference.len() - refused.feasible();
        let representation = match request {
            RowRequest::Full => Representation::Full,
            RowRequest::Viable => Representation::Viable {
                span: source_slack
                    .saturating_add(reference_slack)
                    .min(reference.len()),
            },
        };
        let geometry = match Geometry::for_representation(
            refused.graph(),
            reference.len(),
            refused.preparation(),
            representation,
            control,
        )? {
            Ok(value) => value,
            Err(refusal) => return Ok(Err(refusal)),
        };
        let rows = match request {
            RowRequest::Full => AdmittedRows::Full(FullRows { geometry }),
            RowRequest::Viable => AdmittedRows::Viable(ViableRows {
                geometry,
                source_slack,
                reference_slack,
            }),
        };
        Ok(Ok(Self {
            refused,
            source,
            reference,
            mode,
            rows,
        }))
    }
    pub(super) fn refused(&self) -> RefusedBand<'g, 's> {
        self.refused
    }
    pub(super) fn source(&self) -> &'g [&'s str] {
        self.source
    }
    pub(super) fn reference(&self) -> &'s [String] {
        self.reference
    }
    pub(super) fn mode(&self) -> LiteralMatchMode {
        self.mode
    }
    pub(super) fn geometry(&self) -> &Geometry {
        match &self.rows {
            AdmittedRows::Full(rows) => &rows.geometry,
            AdmittedRows::Viable(rows) => &rows.geometry,
        }
    }
    #[cfg(test)]
    pub(super) fn geometry_mut(&mut self) -> &mut Geometry {
        match &mut self.rows {
            AdmittedRows::Full(rows) => &mut rows.geometry,
            AdmittedRows::Viable(rows) => &mut rows.geometry,
        }
    }
    pub(super) fn source_slack(&self) -> usize {
        match &self.rows {
            AdmittedRows::Full(_) => self.source.len(),
            AdmittedRows::Viable(rows) => rows.source_slack,
        }
    }
    pub(super) fn reference_slack(&self) -> usize {
        match &self.rows {
            AdmittedRows::Full(_) => self.reference.len(),
            AdmittedRows::Viable(rows) => rows.reference_slack,
        }
    }
    /// Full row passes are already reserved; only candidate visits add work.
    pub(super) fn additional_pass_work(&self, units: usize) -> usize {
        match &self.rows {
            AdmittedRows::Full(_) => 0,
            AdmittedRows::Viable(_) => units,
        }
    }
    pub(super) fn report<C: WorkControl + ?Sized>(&self, outer: &C, receipt: &PackedSearchCost) {
        match &self.rows {
            AdmittedRows::Full(_) => {}
            AdmittedRows::Viable(_) => outer.packed_search_cost(receipt),
        }
    }
}
