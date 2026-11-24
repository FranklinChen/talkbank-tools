//! Fixed budgets and progress-rearmed loading windows have distinct owners.
use super::{Deadline, LoadingStall, LoadingWait, ResponseWait, error};
use crate::types::worker_v2::{LoadProgressV2, ProgressEventV2};

/// Callers can await or observe a running operation without replacing its deadline.
#[derive(Debug)]
pub struct RunningWait(RunningKind);
#[derive(Debug)]
enum RunningKind {
    Budget(FixedBudget),
    Loading(LoadingWindow),
}
/// A fixed budget has no rearming operation.
#[derive(Debug)]
struct FixedBudget {
    deadline: Deadline,
    waited_for: error::WorkerWait,
    limit: crate::api::PositiveSeconds,
}
/// Only a loading owner retains the evidence that can renew its own window.
#[derive(Debug)]
struct LoadingWindow {
    deadline: Deadline,
    waited_for: LoadingWait,
    stall: LoadingStall,
    seen: LoadAdvances,
}
impl LoadingWindow {
    fn observe(&mut self, load: &LoadProgressV2) {
        if self.seen.advanced(load) {
            self.deadline = Deadline::after(self.stall.window());
        }
    }
}
/// Every stored stage carries actual evidence. A stage cannot contain an
/// empty observation set or a zero download mark.
#[derive(Debug, Clone, Copy)]
enum StageEvidence {
    Download(std::num::NonZeroU32),
    Phases(ObservedPhases),
    Mixed {
        downloaded: std::num::NonZeroU32,
        phases: ObservedPhases,
    },
}
#[derive(Debug, Clone, Copy)]
enum ObservedPhases {
    BeginOnly,
    CompleteOnly,
    Both,
}
#[derive(Clone, Copy)]
enum LoadEvidence {
    Growth(std::num::NonZeroU32),
    Phase(crate::types::worker_v2::PhaseEdgeV2),
}
impl LoadEvidence {
    fn admit(measure: &crate::types::worker_v2::LoadMeasureV2) -> Option<Self> {
        use crate::types::worker_v2::LoadMeasureV2;
        match measure {
            LoadMeasureV2::DownloadGrowth { grown_kib } => {
                std::num::NonZeroU32::new(*grown_kib).map(Self::Growth)
            }
            LoadMeasureV2::Phase { edge } => Some(Self::Phase(*edge)),
        }
    }
}
impl ObservedPhases {
    const fn first(edge: crate::types::worker_v2::PhaseEdgeV2) -> Self {
        use crate::types::worker_v2::PhaseEdgeV2;
        match edge {
            PhaseEdgeV2::Begin => Self::BeginOnly,
            PhaseEdgeV2::Complete => Self::CompleteOnly,
        }
    }
    const fn record(self, edge: crate::types::worker_v2::PhaseEdgeV2) -> (Self, bool) {
        use crate::types::worker_v2::PhaseEdgeV2;
        match (self, edge) {
            (Self::BeginOnly, PhaseEdgeV2::Begin)
            | (Self::CompleteOnly, PhaseEdgeV2::Complete)
            | (Self::Both, PhaseEdgeV2::Begin | PhaseEdgeV2::Complete) => (self, false),
            (Self::BeginOnly, PhaseEdgeV2::Complete) | (Self::CompleteOnly, PhaseEdgeV2::Begin) => {
                (Self::Both, true)
            }
        }
    }
}
impl StageEvidence {
    const fn first(evidence: LoadEvidence) -> Self {
        match evidence {
            LoadEvidence::Growth(mark) => Self::Download(mark),
            LoadEvidence::Phase(edge) => Self::Phases(ObservedPhases::first(edge)),
        }
    }
    /// Pure facts can be copied for this consuming transition. They carry no
    /// clock, process, lease or authority to rearm another operation.
    fn record(self, evidence: LoadEvidence) -> (Self, bool) {
        match (self, evidence) {
            (Self::Download(mark), LoadEvidence::Growth(observed)) => {
                (Self::Download(mark.max(observed)), observed > mark)
            }
            (Self::Phases(phases), LoadEvidence::Phase(edge)) => {
                let (phases, advanced) = phases.record(edge);
                (Self::Phases(phases), advanced)
            }
            (Self::Mixed { downloaded, phases }, LoadEvidence::Growth(observed)) => (
                Self::Mixed {
                    downloaded: downloaded.max(observed),
                    phases,
                },
                observed > downloaded,
            ),
            (Self::Mixed { downloaded, phases }, LoadEvidence::Phase(edge)) => {
                let (phases, advanced) = phases.record(edge);
                (Self::Mixed { downloaded, phases }, advanced)
            }
            (Self::Download(downloaded), LoadEvidence::Phase(edge)) => (
                Self::Mixed {
                    downloaded,
                    phases: ObservedPhases::first(edge),
                },
                true,
            ),
            (Self::Phases(phases), LoadEvidence::Growth(downloaded)) => {
                (Self::Mixed { downloaded, phases }, true)
            }
        }
    }
}

/// One carried value per stage retains both kinds of admitted observations.
/// Repeated progress performs one lookup and allocates no replacement key.
#[derive(Debug, Default)]
struct LoadAdvances {
    stages: std::collections::HashMap<String, StageEvidence>,
}
impl LoadAdvances {
    fn advanced(&mut self, load: &LoadProgressV2) -> bool {
        let Some(evidence) = LoadEvidence::admit(&load.measure) else {
            return false;
        };
        match self.stages.get_mut(&load.stage) {
            Some(known) => {
                let (updated, advanced) = known.record(evidence);
                *known = updated;
                advanced
            }
            None => {
                self.stages
                    .insert(load.stage.clone(), StageEvidence::first(evidence));
                true
            }
        }
    }
}

impl RunningWait {
    pub(crate) fn start(wait: ResponseWait) -> Self {
        Self(match wait {
            ResponseWait::Budget { waited_for, limit } => RunningKind::Budget(FixedBudget {
                deadline: Deadline::after(limit),
                waited_for,
                limit,
            }),
            ResponseWait::Loading { waited_for, stall } => RunningKind::Loading(LoadingWindow {
                deadline: Deadline::after(stall.window()),
                waited_for,
                stall,
                seen: LoadAdvances::default(),
            }),
        })
    }
    const fn deadline(&self) -> &Deadline {
        match &self.0 {
            RunningKind::Budget(budget) => &budget.deadline,
            RunningKind::Loading(window) => &window.deadline,
        }
    }
    pub(crate) async fn within<F: std::future::Future>(
        &self,
        work: F,
    ) -> Result<F::Output, error::WorkerError> {
        self.deadline()
            .within(work)
            .await
            .ok_or_else(|| self.expiration_error())
    }
    pub(crate) fn observe(&mut self, event: &ProgressEventV2) {
        match event {
            ProgressEventV2::Load(load) => self.observe_load(load),
            ProgressEventV2::Request(_) => {}
        }
    }
    pub(crate) fn observe_load(&mut self, load: &LoadProgressV2) {
        if let RunningKind::Loading(window) = &mut self.0 {
            window.observe(load);
        }
    }
    fn expiration_error(&self) -> error::WorkerError {
        match &self.0 {
            RunningKind::Budget(budget) => error::WorkerError::Timeout {
                waited_for: budget.waited_for.clone(),
                limit: budget.limit,
            },
            RunningKind::Loading(window) => error::WorkerError::Stalled {
                waited_for: window.waited_for,
                stall: window.stall,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::worker_v2::{
        LoadMeasureV2, PhaseEdgeV2, RequestProgressV2, UnitProgressV2, WorkerRequestIdV2,
    };
    use std::time::Duration;
    #[derive(Clone, Copy, Debug)]
    enum WaitMode {
        Loading,
        Fixed,
    }
    const WINDOW: crate::api::PositiveSeconds = crate::api::PositiveSeconds::literal::<120>();
    fn growth(stage: &str, grown_kib: u32) -> LoadProgressV2 {
        LoadProgressV2 {
            stage: stage.into(),
            measure: LoadMeasureV2::DownloadGrowth { grown_kib },
        }
    }
    fn phase(stage: &str, edge: PhaseEdgeV2) -> LoadProgressV2 {
        LoadProgressV2 {
            stage: stage.into(),
            measure: LoadMeasureV2::Phase { edge },
        }
    }
    /// Guards: actual progress values renew a load only on new per-stage growth or a first phase edge, including repeat, zero and stage-isolation cases.
    /// Kind: policy
    /// Not a type: incoming progress bytes may repeat or move backwards. The choice of which concrete numeric or phase evidence advances a live load remains policy after deadline owners are separated.
    /// State: pure progress values exercise the shared observer with zero clocks, processes or network effects.
    /// Concurrency: none
    #[test]
    fn load_observations_admit_only_new_evidence() {
        let mut seen = LoadAdvances::default();
        for (event, advanced) in [
            (growth("a", 0), false),
            (growth("a", 1024), true),
            (growth("a", 2048), true),
            (growth("a", 2048), false),
            (growth("a", 1024), false),
            (growth("b", 1024), true),
            (phase("a", PhaseEdgeV2::Begin), true),
            (phase("a", PhaseEdgeV2::Complete), true),
            (phase("b", PhaseEdgeV2::Begin), true),
            (phase("b", PhaseEdgeV2::Complete), true),
            (phase("a", PhaseEdgeV2::Begin), false),
            (phase("a", PhaseEdgeV2::Complete), false),
            (phase("c", PhaseEdgeV2::Complete), true),
            (phase("c", PhaseEdgeV2::Complete), false),
            (phase("c", PhaseEdgeV2::Begin), true),
            (growth("c", 1), true),
            (phase("d", PhaseEdgeV2::Begin), true),
            (phase("d", PhaseEdgeV2::Begin), false),
            (growth("d", 0), false),
            (growth("d", 4), true),
        ] {
            assert_eq!(seen.advanced(&event), advanced, "{event:?}");
        }
    }
    /// Guards: controlled Tokio time observes renewal only for actual loading progress, fixed budgets keep their original deadline and request counts extend neither owner.
    /// Kind: contract
    /// Not a type: the timer adapter must observe the actual chosen deadline and external progress category. Opaque owners prevent arbitrary deadline replacement but cannot statically certify Tokio clock effects.
    /// State: one paused runtime owns actual timer waits with declared load and request progress, requiring no model, child or socket.
    /// Concurrency: test-local paused time and owned waits isolate every case without wall-clock sleeps or shared state.
    #[tokio::test(start_paused = true)]
    async fn clock_effects_follow_their_fixed_or_loading_owner() {
        let request = ProgressEventV2::Request(
            RequestProgressV2::new(
                WorkerRequestIdV2::from("declared-request"),
                "work".into(),
                UnitProgressV2::new(1, 10).unwrap(),
            )
            .unwrap(),
        );
        for (mode, events, expected_left) in [
            (
                WaitMode::Loading,
                vec![
                    ProgressEventV2::Load(growth("a", 1024)),
                    ProgressEventV2::Load(phase("a", PhaseEdgeV2::Begin)),
                ],
                120,
            ),
            (
                WaitMode::Fixed,
                vec![
                    ProgressEventV2::Load(growth("a", 1024)),
                    ProgressEventV2::Load(phase("a", PhaseEdgeV2::Begin)),
                ],
                20,
            ),
            (
                WaitMode::Loading,
                vec![request.clone(), request.clone()],
                20,
            ),
            (WaitMode::Fixed, vec![request.clone(), request], 20),
        ] {
            let definition = match mode {
                WaitMode::Loading => ResponseWait::Loading {
                    waited_for: LoadingWait::Ready,
                    stall: LoadingStall::new(WINDOW),
                },
                WaitMode::Fixed => ResponseWait::Budget {
                    waited_for: error::WorkerWait::Capabilities,
                    limit: WINDOW,
                },
            };
            let mut running = RunningWait::start(definition);
            for event in events {
                tokio::time::advance(Duration::from_secs(50)).await;
                running.observe(&event);
            }
            let from = tokio::time::Instant::now();
            let expired = running
                .within(std::future::pending::<()>())
                .await
                .expect_err("the pending effect ends only at its owned deadline");
            match (mode, expired) {
                (WaitMode::Loading, error::WorkerError::Stalled { .. })
                | (WaitMode::Fixed, error::WorkerError::Timeout { .. }) => {}
                other => panic!("expiry must retain its operation owner: {other:?}"),
            }
            assert_eq!(from.elapsed(), Duration::from_secs(expected_left));
        }
    }
}
