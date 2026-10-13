//! Shared input-bound packed proof. Admitted full and viable row states retain
//! their distinct geometry and accounting; neither can escape a partial proof.

use super::profiles::{Profile, Row};
use super::*;
pub(super) mod cost;
mod proof;
use cost::{ForwardWork, Ledger, ReconstructionWork};

mod admission;
use admission::{AdmittedProblem, RowRequest};

struct Plan<'g, 's> {
    admitted: AdmittedProblem<'g, 's>,
    forward: LiteralReference<'s>,
    backward: LiteralReference<'s>,
}
impl<'g, 's> Plan<'g, 's> {
    fn graph(&self) -> &Graph {
        self.admitted.refused().graph()
    }
    fn source(&self) -> &'g [&'s str] {
        self.admitted.source()
    }
    fn length(&self) -> usize {
        self.admitted.reference().len()
    }
    fn source_slack(&self) -> usize {
        self.admitted.source_slack()
    }
    fn reference_slack(&self) -> usize {
        self.admitted.reference_slack()
    }
    fn geometry(&self) -> &Geometry {
        self.admitted.geometry()
    }
    fn admit<C: WorkControl + ?Sized>(
        refused: RefusedBand<'g, 's>,
        request: RowRequest,
        control: &C,
    ) -> Result<Result<Self, LocalInterleavingRefusal>, C::Error> {
        let admitted = match AdmittedProblem::observe(refused, request, control)? {
            Ok(value) => value,
            Err(refusal) => return Ok(Err(refusal)),
        };
        let forward = LiteralReference::observe(
            admitted.reference(),
            admitted.mode(),
            ReferenceOrder::Forward,
            control,
        )?;
        let backward = LiteralReference::observe(
            admitted.reference(),
            admitted.mode(),
            ReferenceOrder::Reverse,
            control,
        )?;
        Ok(Ok(Self {
            admitted,
            forward,
            backward,
        }))
    }
}

enum PrefixDirection {}
enum SuffixDirection {}
trait Direction {
    fn edges(graph: &Graph, node: usize) -> &[Edge];
    fn expiry(graph: &Graph, node: usize, slot: usize) -> usize;
    fn required(plan: &Plan<'_, '_>, node: usize) -> usize;
    fn reference<'p, 'g, 's>(plan: &'p Plan<'g, 's>) -> &'p LiteralReference<'s>;
}
impl Direction for PrefixDirection {
    fn edges(graph: &Graph, node: usize) -> &[Edge] {
        &graph.incoming[node]
    }
    fn expiry(graph: &Graph, node: usize, slot: usize) -> usize {
        graph.edges[node]
            .iter()
            .map(|edge| graph.rank[edge.to])
            .max()
            .unwrap_or(slot)
    }
    fn required(plan: &Plan<'_, '_>, node: usize) -> usize {
        plan.graph().depth[node].saturating_sub(plan.source_slack())
    }
    fn reference<'p, 'g, 's>(plan: &'p Plan<'g, 's>) -> &'p LiteralReference<'s> {
        &plan.forward
    }
}
impl Direction for SuffixDirection {
    fn edges(graph: &Graph, node: usize) -> &[Edge] {
        &graph.edges[node]
    }
    fn expiry(graph: &Graph, node: usize, slot: usize) -> usize {
        graph.incoming[node]
            .iter()
            .map(|edge| graph.rank[edge.to])
            .min()
            .unwrap_or(slot)
    }
    fn required(plan: &Plan<'_, '_>, node: usize) -> usize {
        (plan.source().len() - plan.graph().depth[node]).saturating_sub(plan.source_slack())
    }
    fn reference<'p, 'g, 's>(plan: &'p Plan<'g, 's>) -> &'p LiteralReference<'s> {
        &plan.backward
    }
}

struct Frontier<'p, 'g, 's, D> {
    plan: &'p Plan<'g, 's>,
    rows: HashMap<usize, Row>,
    expiry: BTreeMap<usize, Vec<usize>>,
    direction: PhantomData<D>,
}
impl<'p, 'g, 's, D: Direction> Frontier<'p, 'g, 's, D> {
    fn new(plan: &'p Plan<'g, 's>) -> Self {
        Self {
            plan,
            rows: HashMap::new(),
            expiry: BTreeMap::new(),
            direction: PhantomData,
        }
    }
    fn compute<C: PackedControl + ?Sized>(
        &self,
        slot: usize,
        control: &C,
    ) -> Result<Row, C::Error> {
        control.check()?;
        let graph = self.plan.graph();
        let node = graph.order[slot];
        let edges = D::edges(graph, node);
        if edges.is_empty() {
            let span = self.plan.reference_slack();
            control.charge(span.div_ceil(64) + 1)?;
            control.profile_width(span.div_ceil(64));
            return Ok(Row::Viable(Profile::zero(span)));
        }
        let mut result = Row::Unreachable;
        for edge in edges {
            let parent = graph.rank[edge.to];
            debug_assert!(
                self.rows.contains_key(&parent),
                "all live predecessor states are retained"
            );
            let Some(Row::Viable(previous)) = self.rows.get(&parent) else {
                continue;
            };
            control.charge(previous.bits().len() + 1)?;
            let candidate = if let Some(word) = edge.word {
                previous.clone().advance(
                    D::reference(self.plan),
                    self.plan.source()[word],
                    D::required(self.plan, node),
                    self.plan.reference_slack(),
                    self.plan.length(),
                    control,
                )?
            } else {
                Row::Viable(previous.clone())
            };
            result = match (result, candidate) {
                (Row::Unreachable, other) | (other, Row::Unreachable) => other,
                (Row::Viable(a), Row::Viable(b)) => Row::Viable(a.merge(b, control)?),
            };
        }
        if let Row::Viable(profile) = &result {
            control.profile_width(profile.bits().len());
        }
        Ok(result)
    }
    fn commit(&mut self, slot: usize, row: Row) -> Vec<usize> {
        let graph = self.plan.graph();
        let expiry = D::expiry(graph, graph.order[slot], slot);
        self.rows.insert(slot, row);
        self.expiry.entry(expiry).or_default().push(slot);
        let expired = self.expiry.remove(&slot).unwrap_or_default();
        for slot in &expired {
            self.rows.remove(slot);
        }
        expired
    }
}

type Snapshot = Vec<(usize, Row)>;
struct CompleteForward<'l, 'p, 'g, 's, 'c, C: ?Sized> {
    ledger: &'l Ledger<'p, 'g, 's, 'c, C>,
    snapshots: Vec<Snapshot>,
    optimum: usize,
}
struct Block<'o, 'l, 'p, 'g, 's, 'c, C: ?Sized> {
    owner: &'o CompleteForward<'l, 'p, 'g, 's, 'c, C>,
    start: usize,
    rows: Vec<Row>,
    summaries: Vec<Vec<usize>>,
    root_summaries: Vec<Vec<usize>>,
}
fn row_summary(row: &Row) -> Vec<usize> {
    match row {
        Row::Unreachable => Vec::new(),
        Row::Viable(profile) => super::summary(profile.bits()),
    }
}
impl<C: WorkControl + ?Sized> Block<'_, '_, '_, '_, '_, '_, C> {
    fn root(&self, slot: usize) -> Option<usize> {
        self.owner.snapshots[self.start / self.owner.ledger.plan().geometry().block]
            .binary_search_by_key(&slot, |(slot, _)| *slot)
            .ok()
    }
    fn score(&self, slot: usize, k: usize) -> Option<usize> {
        let (row, summary) = if slot >= self.start {
            (
                self.rows.get(slot - self.start)?,
                self.summaries.get(slot - self.start)?,
            )
        } else {
            let i = self.root(slot)?;
            (
                &self.owner.snapshots[self.start / self.owner.ledger.plan().geometry().block][i].1,
                &self.root_summaries[i],
            )
        };
        let Row::Viable(profile) = row else {
            return None;
        };
        if k < profile.first() || k > profile.last() {
            return None;
        }
        Some(profile.base() + super::before(profile.bits(), summary, k - profile.first()))
    }
}
impl<'l, 'p, 'g, 's, 'c, C: WorkControl + ?Sized> CompleteForward<'l, 'p, 'g, 's, 'c, C> {
    fn observe(ledger: &'l Ledger<'p, 'g, 's, 'c, C>) -> Result<Self, Stop<C::Error>> {
        let plan = ledger.plan();
        let control = ledger.stage::<ForwardWork>();
        let mut frontier = Frontier::<PrefixDirection>::new(plan);
        let mut snapshots = Vec::new();
        let mut optimum = 0;
        for slot in 0..plan.graph().order.len() {
            control.check()?;
            if slot % plan.geometry().block == 0 {
                let mut snapshot = Vec::with_capacity(frontier.rows.len());
                for (&slot, row) in &frontier.rows {
                    control.charge(match row {
                        Row::Unreachable => 1,
                        Row::Viable(profile) => profile.bits().len() + 1,
                    })?;
                    snapshot.push((slot, row.clone()));
                }
                snapshot.sort_unstable_by_key(|(slot, _)| *slot);
                snapshots.push(snapshot);
            }
            let row = frontier.compute(slot, &control)?;
            if plan.graph().order[slot] == plan.graph().end
                && let Row::Viable(profile) = &row
            {
                control.charge(profile.bits().len() + 1)?;
                optimum = profile.score(plan.length()).unwrap_or(0);
            }
            let _ = frontier.commit(slot, row);
        }
        debug_assert!(optimum >= plan.admitted.refused().feasible());
        Ok(Self {
            ledger,
            snapshots,
            optimum,
        })
    }
    fn reconstruct(
        &self,
        index: usize,
    ) -> Result<Block<'_, 'l, 'p, 'g, 's, 'c, C>, Stop<C::Error>> {
        let control = self.ledger.stage::<ReconstructionWork>();
        let start = index * self.ledger.plan().geometry().block;
        let end = (start + self.ledger.plan().geometry().block)
            .min(self.ledger.plan().graph().order.len());
        let mut frontier = Frontier::<PrefixDirection>::new(self.ledger.plan());
        let mut root_summaries = Vec::new();
        for (slot, row) in &self.snapshots[index] {
            control.charge(match row {
                Row::Unreachable => 1,
                Row::Viable(profile) => profile.bits().len().saturating_mul(2) + 1,
            })?;
            root_summaries.push(row_summary(row));
            frontier.rows.insert(*slot, row.clone());
            let graph = self.ledger.plan().graph();
            let expiry = PrefixDirection::expiry(graph, graph.order[*slot], *slot);
            frontier.expiry.entry(expiry).or_default().push(*slot);
        }
        let mut rows = Vec::with_capacity(end - start);
        let mut summaries = Vec::with_capacity(end - start);
        for slot in start..end {
            let row = frontier.compute(slot, &control)?;
            control.charge(match &row {
                Row::Unreachable => 1,
                Row::Viable(profile) => profile.bits().len().saturating_mul(2) + 1,
            })?;
            summaries.push(row_summary(&row));
            let _ = frontier.commit(slot, row.clone());
            rows.push(row);
        }
        Ok(Block {
            owner: self,
            start,
            rows,
            summaries,
            root_summaries,
        })
    }
    fn finish(self) -> Result<Result<Observation, LocalInterleavingRefusal>, Stop<C::Error>> {
        proof::observe(self)
    }
}

pub(super) fn observe<C: WorkControl + ?Sized>(
    refused: RefusedBand<'_, '_>,
    control: &C,
) -> Result<Result<Observation, LocalInterleavingRefusal>, C::Error> {
    // Preserve the previous full-profile admission/fallback policy exactly.
    if let Relation::Literal { source, .. } = refused.relation()
        && source.len() >= refused.feasible().saturating_mul(2)
        && let Ok(plan) = Plan::admit(refused, RowRequest::Full, control)?
    {
        return execute(&plan, control);
    }
    let plan = match Plan::admit(refused, RowRequest::Viable, control)? {
        Ok(plan) => plan,
        Err(refusal) => return Ok(Err(refusal)),
    };
    execute(&plan, control)
}

fn execute<C: WorkControl + ?Sized>(
    plan: &Plan<'_, '_>,
    control: &C,
) -> Result<Result<Observation, LocalInterleavingRefusal>, C::Error> {
    let ledger = Ledger::new(plan, control);
    let result = CompleteForward::observe(&ledger).and_then(|forward| forward.finish());
    let status = match &result {
        Ok(Ok(_)) => PackedSearchStatus::Completed,
        Ok(Err(refusal)) | Err(Stop::Refused(refusal)) => PackedSearchStatus::Refused(*refusal),
        Err(Stop::Cancelled(_)) => PackedSearchStatus::Interrupted,
    };
    ledger.report(status);
    match result {
        Ok(result) => Ok(result),
        Err(Stop::Cancelled(error)) => Err(error),
        Err(Stop::Refused(refusal)) => Ok(Err(refusal)),
    }
}

#[cfg(test)]
pub(super) fn testing<C: WorkControl + ?Sized>(
    refused: RefusedBand<'_, '_>,
    block: usize,
    control: &C,
) -> Result<Result<Observation, LocalInterleavingRefusal>, C::Error> {
    let mut plan = match Plan::admit(refused, RowRequest::Viable, control)? {
        Ok(plan) => plan,
        Err(refusal) => return Ok(Err(refusal)),
    };
    plan.admitted.geometry_mut().block = block;
    execute(&plan, control)
}

#[cfg(test)]
pub(super) fn testing_full<C: WorkControl + ?Sized>(
    refused: RefusedBand<'_, '_>,
    block: usize,
    control: &C,
) -> Result<Result<Observation, LocalInterleavingRefusal>, C::Error> {
    let mut plan = match Plan::admit(refused, RowRequest::Full, control)? {
        Ok(plan) => plan,
        Err(refusal) => return Ok(Err(refusal)),
    };
    plan.admitted.geometry_mut().block = block;
    execute(&plan, control)
}
