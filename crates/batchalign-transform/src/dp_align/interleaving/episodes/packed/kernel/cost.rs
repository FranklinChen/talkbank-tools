//! Input-bound work ledger. Only stage scopes can charge work; no
//! consumer can construct a receipt or convert its costs into path evidence.

use super::*;
use std::cell::Cell;

/// Classification of charged work, not a proof-completion state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackedSearchStage {
    /// Witness/preparation and prospective graph metadata reserves.
    Preparation,
    /// Initial forward scores and retained checkpoints.
    Forward,
    /// Forward block reconstruction during backward proof.
    Reconstruction,
    /// Reverse scoring and suffix profiles.
    Backward,
    /// All-optimal edge cuts and addressed candidate enumeration.
    Edges,
    /// Inspection probes over live proof blocks, reserved in preparation.
    Traceback,
}
impl PackedSearchStage {
    /// Stable diagnostic names; no new budget or matching policy.
    pub const ALL: [Self; 6] = [
        Self::Preparation,
        Self::Forward,
        Self::Reconstruction,
        Self::Backward,
        Self::Edges,
        Self::Traceback,
    ];
    /// Diagnostic spelling.
    pub fn name(self) -> &'static str {
        match self {
            Self::Preparation => "preparation",
            Self::Forward => "forward",
            Self::Reconstruction => "reconstruction",
            Self::Backward => "backward",
            Self::Edges => "edges",
            Self::Traceback => "traceback",
        }
    }
}
/// Search outcome, never permission to inject timing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackedSearchStatus {
    /// The complete search returned an observation.
    Completed,
    /// A named finite budget prevented completion.
    Refused(LocalInterleavingBudget),
    /// The search contradicted its own admission and refused; not a budget.
    Inconsistent(LocalInterleavingInconsistency),
    /// The caller interrupted computation; no observation was returned.
    Interrupted,
}
impl PackedSearchStatus {
    /// Diagnostic spelling.
    pub fn name(self) -> &'static str {
        match self {
            Self::Completed => "completed",
            Self::Refused(_) => "refused",
            Self::Inconsistent(_) => "inconsistent",
            Self::Interrupted => "interrupted",
        }
    }
}

/// Immutable costs of one actual admitted search. Construction belongs to the
/// graph producer; these values do not certify correspondence or acoustics.
#[derive(Debug, Clone)]
pub struct PackedSearchCost {
    status: PackedSearchStatus,
    last_stage: PackedSearchStage,
    source_words: usize,
    reference_words: usize,
    graph_nodes: usize,
    checkpoint_stride: usize,
    reserved_score_units: usize,
    profile_width_bound: usize,
    peak_profile_words: usize,
    charged: [usize; 6],
    required_at_least: Option<usize>,
}
impl PackedSearchCost {
    /// Observed completion or refusal, not a certificate.
    pub fn status(&self) -> PackedSearchStatus {
        self.status
    }
    /// Stage of the last checkpoint/operation, including a denied operation.
    pub fn last_stage(&self) -> PackedSearchStage {
        self.last_stage
    }
    /// Projected source population.
    pub fn source_words(&self) -> usize {
        self.source_words
    }
    /// Projected reference population.
    pub fn reference_words(&self) -> usize {
        self.reference_words
    }
    /// Actual source graph population.
    pub fn graph_nodes(&self) -> usize {
        self.graph_nodes
    }
    /// Admitted checkpoint stride.
    pub fn checkpoint_stride(&self) -> usize {
        self.checkpoint_stride
    }
    /// Admitted conservative storage payload, not process RSS.
    pub fn reserved_score_units(&self) -> usize {
        self.reserved_score_units
    }
    /// Geometric maximum packed words per viable row.
    pub fn profile_width_bound(&self) -> usize {
        self.profile_width_bound
    }
    /// Maximum width of rows actually produced in the observed prefix.
    pub fn peak_profile_words(&self) -> usize {
        self.peak_profile_words
    }
    /// Reserved/charged work by stage; denied work is not included.
    pub fn charged(&self, stage: PackedSearchStage) -> usize {
        self.charged[stage as usize]
    }
    /// Total charged work. Preparation includes prospective metadata reserves.
    pub fn used(&self) -> usize {
        self.charged.iter().sum()
    }
    /// The next denied operation's known floor, not the eventual full-run cost.
    pub fn required_at_least(&self) -> Option<usize> {
        self.required_at_least
    }
}

pub(super) trait Stage {
    const ID: PackedSearchStage;
}
macro_rules! stages { ($($name:ident => $id:ident),*) => { $(pub(super) enum $name {} impl Stage for $name {const ID:PackedSearchStage=PackedSearchStage::$id;})* }; }
stages!(ForwardWork=>Forward, ReconstructionWork=>Reconstruction, BackwardWork=>Backward, EdgeWork=>Edges, TracebackWork=>Traceback);

pub(super) struct Ledger<'p, 'g, 's, 'c, C: ?Sized> {
    plan: &'p Plan<'g, 's>,
    outer: &'c C,
    used: Cell<usize>,
    charged: [Cell<usize>; 6],
    last: Cell<PackedSearchStage>,
    peak: Cell<usize>,
    denied: Cell<Option<usize>>,
}
pub(super) struct Scope<'a, 'p, 'g, 's, 'c, C: ?Sized, S> {
    ledger: &'a Ledger<'p, 'g, 's, 'c, C>,
    stage: PhantomData<S>,
}
impl<'p, 'g, 's, 'c, C: WorkControl + ?Sized> Ledger<'p, 'g, 's, 'c, C> {
    pub(super) fn new(plan: &'p Plan<'g, 's>, outer: &'c C) -> Self {
        let charged = std::array::from_fn(|_| Cell::new(0));
        charged[PackedSearchStage::Preparation as usize].set(plan.geometry().work);
        Self {
            plan,
            outer,
            used: Cell::new(plan.geometry().work),
            charged,
            last: Cell::new(PackedSearchStage::Preparation),
            peak: Cell::new(0),
            denied: Cell::new(None),
        }
    }
    pub(super) fn plan(&self) -> &'p Plan<'g, 's> {
        self.plan
    }
    fn account<S: Stage>(&self, units: usize) -> Result<(), Stop<C::Error>> {
        self.last.set(S::ID);
        let required = self.used.get().saturating_add(units);
        if required > WORK_CELLS {
            self.denied.set(Some(required));
            return Err(Stop::Refused(LocalInterleavingRefusal::Budget(
                work_refusal(required),
            )));
        }
        self.used.set(required);
        let charged = &self.charged[S::ID as usize];
        charged.set(charged.get() + units);
        Ok(())
    }
    pub(super) fn stage<S: Stage>(&self) -> Scope<'_, 'p, 'g, 's, 'c, C, S> {
        Scope {
            ledger: self,
            stage: PhantomData,
        }
    }
    pub(super) fn report(self, status: PackedSearchStatus) {
        let receipt = PackedSearchCost {
            status,
            last_stage: self.last.get(),
            source_words: self.plan.source().len(),
            reference_words: self.plan.length(),
            graph_nodes: self.plan.graph().order.len(),
            checkpoint_stride: self.plan.geometry().block,
            reserved_score_units: self.plan.geometry().cells,
            profile_width_bound: self
                .plan
                .source_slack()
                .saturating_add(self.plan.reference_slack())
                .min(self.plan.length())
                .div_ceil(64),
            peak_profile_words: self.peak.get(),
            charged: std::array::from_fn(|i| self.charged[i].get()),
            required_at_least: self.denied.get(),
        };
        self.plan.admitted.report(self.outer, &receipt);
    }
}
impl<C: WorkControl + ?Sized, S: Stage> WorkControl for Scope<'_, '_, '_, '_, '_, C, S> {
    type Error = Stop<C::Error>;
    fn check(&self) -> Result<(), Self::Error> {
        self.ledger.last.set(S::ID);
        self.ledger.outer.check().map_err(Stop::Cancelled)
    }
}
impl<C: WorkControl + ?Sized, S: Stage> PackedControl for Scope<'_, '_, '_, '_, '_, C, S> {
    fn charge(&self, units: usize) -> Result<(), Self::Error> {
        self.ledger
            .account::<S>(self.ledger.plan.admitted.additional_pass_work(units))
    }
    fn profile_width(&self, words: usize) {
        self.ledger.peak.set(self.ledger.peak.get().max(words));
    }
    fn refuse(&self, inconsistency: LocalInterleavingInconsistency) -> Self::Error {
        Stop::Refused(LocalInterleavingRefusal::Inconsistent(inconsistency))
    }
}

impl<C: WorkControl + ?Sized> CandidateControl for Scope<'_, '_, '_, '_, '_, C, EdgeWork> {
    fn candidate(&self) -> Result<(), Self::Error> {
        self.ledger.account::<EdgeWork>(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{cell::RefCell, convert::Infallible};

    #[derive(Default)]
    struct Recorder(RefCell<Vec<PackedSearchCost>>);
    impl WorkControl for Recorder {
        type Error = Infallible;
        fn check(&self) -> Result<(), Self::Error> {
            Ok(())
        }
        fn packed_search_cost(&self, cost: &PackedSearchCost) {
            self.0.borrow_mut().push(cost.clone());
        }
    }
    fn with_plan(f: impl FnOnce(&Plan<'_, '_>)) {
        let a = vec!["a".to_owned()];
        let b = vec!["b".to_owned()];
        let reference = vec!["b".to_owned(), "a".to_owned()];
        let turns = [SpeakerTurn::new("A", &a), SpeakerTurn::new("B", &b)];
        let (graph, addresses) = Graph::of(&turns, Budget::nodes(&turns).unwrap());
        let relation = Relation::new(
            addresses
                .iter()
                .map(|p| turns[p.utterance].words[p.word].as_str())
                .collect(),
            &reference,
            MatchMode::Exact,
        );
        let plan = uninterrupted(Plan::admit(
            RefusedBand::testing(&graph, &relation),
            RowRequest::Viable,
            &Uninterrupted,
        ))
        .unwrap();
        f(&plan);
    }
    #[test]
    fn scope_receipt_totals_exclude_denied_work_and_keep_its_actual_stage() {
        with_plan(|plan| {
            let recorder = Recorder::default();
            let ledger = Ledger::new(plan, &recorder);
            // A boundary test seeds only the scalar ledger, not graph evidence.
            ledger.used.set(WORK_CELLS - 1);
            ledger.charged[0].set(WORK_CELLS - 1);
            assert!(ledger.stage::<ForwardWork>().charge(1).is_ok());
            assert!(matches!(
                ledger.stage::<EdgeWork>().charge(1),
                Err(Stop::Refused(_))
            ));
            ledger.report(PackedSearchStatus::Refused(work_refusal(WORK_CELLS + 1)));
            let costs = recorder.0.borrow();
            let cost = &costs[0];
            insta::assert_json_snapshot!((cost.status().name(),cost.last_stage().name(),cost.used(),cost.required_at_least(),cost.charged),@r###"
            [
              "refused",
              "edges",
              64000000,
              64000001,
              [
                63999999,
                1,
                0,
                0,
                0,
                0
              ]
            ]
            "###);
        });
    }
    #[test]
    fn completed_cost_receipt_is_input_bound_and_does_not_change_proof() {
        with_plan(|plan| {
            let recorder = Recorder::default();
            let observed = uninterrupted(super::super::execute(plan, &recorder)).unwrap();
            let plain = uninterrupted(super::super::execute(plan, &Uninterrupted)).unwrap();
            assert_eq!(observed.selected.len(), plain.selected.len());
            for (a, b) in observed.words.iter().zip(&plain.words) {
                assert_eq!(a.candidates, b.candidates);
                assert_eq!(a.can_be_missing, b.can_be_missing);
            }
            let costs = recorder.0.borrow();
            assert_eq!(costs.len(), 1);
            let cost = &costs[0];
            assert_eq!(cost.status(), PackedSearchStatus::Completed);
            assert_eq!(cost.required_at_least(), None);
            assert_eq!(cost.source_words(), 2);
            assert_eq!(cost.reference_words(), 2);
            assert_eq!(cost.graph_nodes(), plan.graph().order.len());
            assert_eq!(cost.checkpoint_stride(), plan.geometry().block);
            assert!(cost.peak_profile_words() <= cost.profile_width_bound());
            assert!(cost.charged(PackedSearchStage::Forward) > 0);
            assert!(cost.charged(PackedSearchStage::Reconstruction) > 0);
            assert!(cost.charged(PackedSearchStage::Backward) > 0);
            assert!(cost.charged(PackedSearchStage::Edges) > 0);
        });
    }
    #[test]
    fn completed_proof_reuses_its_blocks_for_inspection_trace() {
        with_plan(|plan| {
            let recorder = Recorder::default();
            let observed = uninterrupted(super::super::execute(plan, &recorder)).unwrap();
            let selected: Vec<_> = observed
                .selected
                .iter()
                .map(|pair| (pair.source, pair.reference))
                .collect();
            assert_eq!(selected, [(1, 0), (0, 1)]);
            let costs = recorder.0.borrow();
            assert_eq!(costs.len(), 1);
            assert_eq!(costs[0].status(), PackedSearchStatus::Completed);
            // Inspection walks live proof blocks; its scalar probes are already
            // reserved in preparation, so it needs no second packed pass.
            assert_eq!(costs[0].charged(PackedSearchStage::Traceback), 0);
        });
    }
    #[test]
    fn full_admission_reserves_passes_but_still_bounds_candidate_visits() {
        with_plan(|viable| {
            let recorder = Recorder::default();
            let full = uninterrupted(Plan::admit(
                viable.admitted.refused(),
                RowRequest::Full,
                &recorder,
            ))
            .unwrap();
            assert_eq!(full.source_slack(), full.source().len());
            assert_eq!(full.reference_slack(), full.length());
            let ledger = Ledger::new(&full, &recorder);
            ledger.used.set(WORK_CELLS - 1);
            ledger.charged[0].set(WORK_CELLS - 1);
            // Already reserved full-row passes cannot be double-charged.
            assert!(ledger.stage::<ForwardWork>().charge(WORK_CELLS).is_ok());
            assert_eq!(ledger.used.get(), WORK_CELLS - 1);
            assert!(ledger.stage::<EdgeWork>().candidate().is_ok());
            assert!(matches!(
                ledger.stage::<EdgeWork>().candidate(),
                Err(Stop::Refused(_))
            ));
            ledger.report(PackedSearchStatus::Refused(work_refusal(WORK_CELLS + 1)));
            assert!(
                recorder.0.borrow().is_empty(),
                "full rows keep the existing observer scope"
            );
        });
    }
    #[test]
    fn cancelled_stage_reports_no_completed_observation() {
        struct Cancel(RefCell<Vec<PackedSearchCost>>);
        impl WorkControl for Cancel {
            type Error = &'static str;
            fn check(&self) -> Result<(), Self::Error> {
                Err("cancelled")
            }
            fn packed_search_cost(&self, cost: &PackedSearchCost) {
                self.0.borrow_mut().push(cost.clone());
            }
        }
        with_plan(|plan| {
            let cancel = Cancel(RefCell::new(Vec::new()));
            assert!(matches!(
                super::super::execute(plan, &cancel),
                Err("cancelled")
            ));
            let costs = cancel.0.borrow();
            assert_eq!(costs.len(), 1);
            assert_eq!(costs[0].status(), PackedSearchStatus::Interrupted);
            assert_eq!(costs[0].last_stage(), PackedSearchStage::Forward);
            assert_eq!(costs[0].charged(PackedSearchStage::Forward), 0);
        });
    }
}
