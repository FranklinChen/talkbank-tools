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

/// Cumulative diagnostic work, never an execution budget or proof capability.
/// Beyond the counter's numeric range, retain an explicit lower bound rather
/// than wrap, invent an exact total, or refuse otherwise valid correspondence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackedSearchWork {
    /// All charged operations are represented exactly.
    Exact(u128),
    /// At least this much work was charged; the exact total exceeded u128.
    AtLeast(u128),
}
impl PackedSearchWork {
    /// Known lower bound on reserved/charged abstract operations, not time.
    pub fn lower_bound(self) -> u128 {
        match self {
            Self::Exact(units) | Self::AtLeast(units) => units,
        }
    }
    fn plus(self, other: Self) -> Self {
        match (self, other) {
            (Self::Exact(a), Self::Exact(b)) => match a.checked_add(b) {
                Some(total) => Self::Exact(total),
                None => Self::AtLeast(u128::MAX),
            },
            (a, b) => Self::AtLeast(a.lower_bound().saturating_add(b.lower_bound())),
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
    charged: [PackedSearchWork; 6],
}
impl PackedSearchCost {
    /// Observed completion or refusal, not a certificate.
    pub fn status(&self) -> PackedSearchStatus {
        self.status
    }
    /// Stage of the last checkpoint/operation.
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
    /// Reserved/charged diagnostic work by stage.
    pub fn charged(&self, stage: PackedSearchStage) -> PackedSearchWork {
        self.charged[stage as usize]
    }
    /// Total charged work. Preparation includes prospective metadata reserves.
    pub fn used(&self) -> PackedSearchWork {
        self.charged
            .iter()
            .copied()
            .fold(PackedSearchWork::Exact(0), PackedSearchWork::plus)
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
    charged: [Cell<PackedSearchWork>; 6],
    last: Cell<PackedSearchStage>,
    peak: Cell<usize>,
}
pub(super) struct Scope<'a, 'p, 'g, 's, 'c, C: ?Sized, S> {
    ledger: &'a Ledger<'p, 'g, 's, 'c, C>,
    stage: PhantomData<S>,
}
impl<'p, 'g, 's, 'c, C: WorkControl + ?Sized> Ledger<'p, 'g, 's, 'c, C> {
    pub(super) fn new(plan: &'p Plan<'g, 's>, outer: &'c C) -> Self {
        let charged = std::array::from_fn(|_| Cell::new(PackedSearchWork::Exact(0)));
        charged[PackedSearchStage::Preparation as usize]
            .set(PackedSearchWork::Exact(plan.geometry().work as u128));
        Self {
            plan,
            outer,
            charged,
            last: Cell::new(PackedSearchStage::Preparation),
            peak: Cell::new(0),
        }
    }
    pub(super) fn plan(&self) -> &'p Plan<'g, 's> {
        self.plan
    }
    /// Counting cannot refuse execution. Cancellation and proof/resource
    /// admission remain separate transitions with their own result types.
    fn account<S: Stage>(&self, units: usize) {
        self.last.set(S::ID);
        let charged = &self.charged[S::ID as usize];
        charged.set(charged.get().plus(PackedSearchWork::Exact(units as u128)));
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
            .account::<S>(self.ledger.plan.admitted.additional_pass_work(units));
        Ok(())
    }
    fn profile_width(&self, words: usize) {
        self.ledger.peak.set(self.ledger.peak.get().max(words));
    }
    fn refuse(&self, inconsistency: LocalInterleavingInconsistency) -> Self::Error {
        Stop::Refused(LocalInterleavingRefusal::Inconsistent(inconsistency))
    }
}

impl<C: WorkControl + ?Sized> Scope<'_, '_, '_, '_, '_, C, EdgeWork> {
    /// Count observed candidate bits in one bounded block. Diagnostic work
    /// cannot refuse, and other stages cannot charge addressed candidates.
    pub(super) fn candidates(&self, count: usize) {
        self.ledger.account::<EdgeWork>(count);
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
    fn with_plan(f: impl FnOnce(Plan<'_, '_>)) {
        let a = vec!["a".to_owned()];
        let b = vec!["b".to_owned()];
        let reference = vec!["b".to_owned(), "a".to_owned()];
        let turns = [SpeakerTurn::new("A", &a), SpeakerTurn::new("B", &b)];
        let materialized = PreparedGraph::admit(&turns)
            .unwrap()
            .materialize(&Uninterrupted)
            .unwrap();
        let graph = materialized.graph();
        let addresses = materialized.addresses();
        let relation = Relation::new(
            addresses
                .iter()
                .map(|p| turns[p.utterance].words[p.word].as_str())
                .collect(),
            &reference,
            MatchMode::Exact,
        );
        let plan = uninterrupted(Plan::admit(
            uninterrupted(SurveyedLiteralSearch::observe(
                PreparedLiteralSearch::testing(graph, &relation),
                &Uninterrupted,
            ))
            .unwrap(),
            RowRequest::Viable,
            &Uninterrupted,
        ))
        .unwrap();
        f(plan);
    }
    #[test]
    fn completed_cost_receipt_is_input_bound() {
        with_plan(|plan| {
            let recorder = Recorder::default();
            let graph_nodes = plan.graph().order.len();
            let checkpoint_stride = plan.geometry().block;
            let _observed = uninterrupted(super::super::execute(plan, &recorder)).unwrap();
            let costs = recorder.0.borrow();
            assert_eq!(costs.len(), 1);
            let cost = &costs[0];
            assert_eq!(cost.status(), PackedSearchStatus::Completed);
            assert_eq!(cost.source_words(), 2);
            assert_eq!(cost.reference_words(), 2);
            assert_eq!(cost.graph_nodes(), graph_nodes);
            assert_eq!(cost.checkpoint_stride(), checkpoint_stride);
            assert!(cost.peak_profile_words() <= cost.profile_width_bound());
            assert!(cost.charged(PackedSearchStage::Forward).lower_bound() > 0);
            assert!(
                cost.charged(PackedSearchStage::Reconstruction)
                    .lower_bound()
                    > 0
            );
            assert!(cost.charged(PackedSearchStage::Backward).lower_bound() > 0);
            assert!(cost.charged(PackedSearchStage::Edges).lower_bound() > 0);
            // Numeric overflow loses exactness explicitly; diagnostics cannot
            // turn a completed correspondence proof into a work refusal.
            assert_eq!(
                PackedSearchWork::Exact(u128::MAX).plus(cost.used()),
                PackedSearchWork::AtLeast(u128::MAX),
            );
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
            assert_eq!(
                costs[0].charged(PackedSearchStage::Traceback),
                PackedSearchWork::Exact(0)
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
            assert_eq!(
                costs[0].charged(PackedSearchStage::Forward),
                PackedSearchWork::Exact(0)
            );
        });
    }
}
