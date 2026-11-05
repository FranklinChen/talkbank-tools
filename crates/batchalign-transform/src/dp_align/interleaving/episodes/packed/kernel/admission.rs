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
    surveyed: SurveyedLiteralSearch<'g, 's>,
    rows: AdmittedRows,
}

/// A rejected representation returns the original preparation owner. It cannot
/// duplicate the paid witness or coexist with an admitted execution plan.
#[derive(Debug)]
pub(super) struct RejectedRows<'g, 's> {
    pub(super) input: SurveyedLiteralSearch<'g, 's>,
    pub(super) cause: LocalInterleavingRefusal,
}

impl<'g, 's> AdmittedProblem<'g, 's> {
    pub(super) fn observe<C: WorkControl + ?Sized>(
        surveyed: SurveyedLiteralSearch<'g, 's>,
        request: RowRequest,
        control: &C,
    ) -> Result<Result<Self, RejectedRows<'g, 's>>, C::Error> {
        let prepared = surveyed.prepared();
        let source = prepared.source();
        let reference = prepared.reference();
        let source_slack = source.len() - prepared.feasible();
        let reference_slack = reference.len() - prepared.feasible();
        let representation = match request {
            RowRequest::Full => Representation::Full,
            RowRequest::Viable => Representation::Viable {
                span: source_slack
                    .saturating_add(reference_slack)
                    .min(reference.len()),
            },
        };
        control.check()?;
        let geometry = match surveyed.project(representation) {
            Ok(value) => value,
            Err(cause) => {
                return Ok(Err(RejectedRows {
                    input: surveyed,
                    cause,
                }));
            }
        };
        let rows = match request {
            RowRequest::Full => AdmittedRows::Full(FullRows { geometry }),
            RowRequest::Viable => AdmittedRows::Viable(ViableRows {
                geometry,
                source_slack,
                reference_slack,
            }),
        };
        Ok(Ok(Self { surveyed, rows }))
    }
    pub(super) fn prepared(&self) -> &PreparedLiteralSearch<'g, 's> {
        self.surveyed.prepared()
    }
    pub(super) fn source(&self) -> &'g [&'s str] {
        self.prepared().source()
    }
    pub(super) fn reference(&self) -> &'s [String] {
        self.prepared().reference()
    }
    pub(super) fn mode(&self) -> LiteralMatchMode {
        self.prepared().mode()
    }
    pub(super) fn geometry(&self) -> &Geometry {
        match &self.rows {
            AdmittedRows::Full(rows) => &rows.geometry,
            AdmittedRows::Viable(rows) => &rows.geometry,
        }
    }
    pub(super) fn source_slack(&self) -> usize {
        match &self.rows {
            AdmittedRows::Full(_) => self.source().len(),
            AdmittedRows::Viable(rows) => rows.source_slack,
        }
    }
    pub(super) fn reference_slack(&self) -> usize {
        match &self.rows {
            AdmittedRows::Full(_) => self.reference().len(),
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
