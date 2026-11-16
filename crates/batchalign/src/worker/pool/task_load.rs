//! The on-demand load lifecycle of one worker, as states.
//!
//! No worker loads a model when it starts. Each model loads when the pool
//! prepares a worker for a request (`ensure_task`), and what a worker knows
//! about one load is a state, not a flag. Every model a worker holds is
//! covered by a host-memory lease, and the lease is part of the state:
//!
//! ```text
//! TaskLoad --LoadAdmission::admit (the host grants its lease; may queue)--> AdmittedLoad
//! AdmittedLoad --the transport's ensure_task round trip,
//!                then AdmittedLoad::confirm (the lease settles)----------> ConfirmedLoad
//! ConfirmedLoad --TaskLoads::record----------------------------------------> Loaded
//! ReadyRequest --LoadPreparation::prepare_on (actual transport and batch lease)--> RunPermit
//! ```
//!
//! - [`TaskLoad`] is one load's identity: the task, the engine overrides its
//!   load reads (the task's own keys, `EngineSelection::task_keys`), so a
//!   second speaker model on one worker is a second load, and its
//!   [`Residency`], what the load keeps resident, derived from the same keys.
//! - [`AdmittedLoad`] is a load whose host-memory lease the coordinator
//!   granted, BEFORE the worker is asked to load. Made only by
//!   [`LoadAdmission::admit`]. A load that fails after admission drops it,
//!   and the lease with it: no lease outlives a failed load.
//! - [`ConfirmedLoad`] is a load the worker answered, its lease settled to
//!   what the resident model has yet to use. Made only by
//!   [`AdmittedLoad::confirm`], so it is settled before any lock is taken to
//!   record it.
//! - [`Loaded`] is the proof that one worker completed that load, and it OWNS
//!   the load's lease: minted only by [`TaskLoads::record`], which consumes
//!   the [`ConfirmedLoad`], so no route records a `Loaded` without the lease,
//!   and the lease is released when the proof is dropped (the worker goes).
//! - [`RunPermit`] is the one input every execute takes: a request paired
//!   with the proof of its own load, and, for a model built per batch
//!   (coreference), the batch's own lease, held for the request.
//! - [`TaskLoads`] holds a worker's `Loaded` states (an absent entry is
//!   `NotLoaded`). A repeated preparation returns the recorded proof without
//!   a round trip and takes no second lease.

use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::api::MemoryMb;
use crate::host_memory::{
    HostMemoryCoordinator, HostMemoryLease, MemoryReclaim, WorkerLeaseRequest,
};
use crate::types::engines::EngineOverrides;
use crate::types::worker_v2::{ExecuteRequestV2, FaRequestMemoryV2};
use crate::worker::error::WorkerError;
use crate::worker::memory_guard::MemoryGuardError;
use crate::worker::residency::Residency;
use crate::worker::{InferTask, LoadingStall, WorkerPid};

use super::EngineSelection;
use super::execute_v2::{execute_request_overrides, infer_task_for_execute_v2};
use super::fa_capacity::{FaHostAssessment, FaPhysicalCapacity, FaProjectedDemand};
use super::fa_preparation::ReadyRequest;

/// One on-demand load: a task, the engine overrides its load reads, and what
/// it keeps resident.
#[derive(Debug, Clone)]
pub struct TaskLoad(Arc<LoadFacts>);

#[derive(Debug)]
struct LoadFacts {
    identity: LoadIdentity,
    selection: EngineSelection,
}

/// Cache identity preserves the pre-existing wire/residency relation. The
/// carried typed selection is a computed artifact with the same private producer.
#[derive(Debug, PartialEq, Eq, Hash)]
struct LoadIdentity {
    task: InferTask,
    overrides: BTreeMap<String, String>,
    residency: Residency,
}

impl PartialEq for TaskLoad {
    fn eq(&self, other: &Self) -> bool {
        self.0.identity == other.0.identity
    }
}
impl Eq for TaskLoad {}
impl Hash for TaskLoad {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.identity.hash(state);
    }
}

impl TaskLoad {
    /// The load `task` needs, given the engines `selected` names.
    pub(super) fn new(task: InferTask, selected: EngineOverrides) -> Self {
        let keys = EngineSelection::task_keys(task, selected);
        Self(Arc::new(LoadFacts {
            identity: LoadIdentity {
                task,
                residency: Residency::of(task, &keys),
                overrides: keys.dispatch_overrides(),
            },
            selection: EngineSelection::from_overrides(keys),
        }))
    }

    /// The load a request needs: its pool task and the engines it names.
    pub fn for_request(request: &ExecuteRequestV2) -> Self {
        let task = infer_task_for_execute_v2(request.task);
        Self::new(task, execute_request_overrides(request))
    }

    /// The task loaded.
    pub fn task(&self) -> InferTask {
        self.0.identity.task
    }

    /// What the load keeps resident in its worker.
    pub fn residency(&self) -> Residency {
        self.0.identity.residency
    }

    /// The overrides the `ensure_task` request carries, `None` when empty.
    pub(crate) fn wire_overrides(&self) -> Option<&BTreeMap<String, String>> {
        (!self.0.identity.overrides.is_empty()).then_some(&self.0.identity.overrides)
    }

    pub(super) fn worker_key(
        &self,
        language: crate::api::WorkerLanguage,
        mode: crate::worker::WorkerBootstrapMode,
    ) -> super::WorkerKey {
        super::WorkerKey {
            target: crate::worker::WorkerTarget::from_infer_task(self.task(), mode),
            language,
            engine_selection: self.0.selection.clone(),
        }
    }

    /// The load as a ledger label names it: the task and its engines.
    fn describe(&self) -> String {
        let mut what = self.task().wire_name().to_owned();
        for (key, value) in &self.0.identity.overrides {
            // Writing to a `String` cannot fail.
            let _ = write!(what, ":{key}={value}");
        }
        what
    }
}

/// Whether a worker hosts models at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelHosting {
    /// A real worker: each load builds what its residency says.
    Models,
    /// A test-echo worker, which answers every load without building
    /// anything, so it holds no model and takes no lease.
    TestEcho,
}

/// What admits a load: the host-memory coordinator the worker's host shares,
/// whether the worker hosts models, and what a queued load retires to make
/// room (the pool's idle workers).
#[derive(Clone)]
pub struct LoadAdmission {
    coordinator: HostMemoryCoordinator,
    hosting: ModelHosting,
    reclaim: Arc<dyn MemoryReclaim>,
}

impl std::fmt::Debug for LoadAdmission {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoadAdmission")
            .field("coordinator", &self.coordinator)
            .field("hosting", &self.hosting)
            .finish_non_exhaustive()
    }
}

impl LoadAdmission {
    #[cfg(test)]
    pub(crate) fn reserved_for_test(&self) -> MemoryMb {
        self.coordinator
            .snapshot()
            .expect("isolated ledger snapshot")
            .active_reserved_mb
    }

    /// Loads admitted by `coordinator` for workers that host as `hosting`; a
    /// load that has to queue retires idle workers through `reclaim`.
    pub fn new(
        coordinator: HostMemoryCoordinator,
        hosting: ModelHosting,
        reclaim: Arc<dyn MemoryReclaim>,
    ) -> Self {
        Self {
            coordinator,
            hosting,
            reclaim,
        }
    }

    /// The `TaskLoad` to `AdmittedLoad` transition: the host grants `load`'s
    /// lease on `worker` before the worker is asked to load.
    ///
    /// Queues while the host cannot fit it (no deadline, reported, see
    /// `HostMemoryCoordinator::queue_for_lease`), so a cancelled job stops
    /// waiting by dropping this future; a load larger than the machine is
    /// refused at once. A load that builds nothing, or any load on a
    /// test-echo worker, is admitted with no lease.
    pub async fn admit(
        &self,
        load: &TaskLoad,
        worker: WorkerPid,
    ) -> Result<AdmittedLoad, WorkerError> {
        let hold = match (self.hosting, load.residency().loading()) {
            (ModelHosting::TestEcho, _) | (ModelHosting::Models, None) => Hold::Unleased,
            (ModelHosting::Models, Some(reserved)) => {
                let request = WorkerLeaseRequest::model_load(worker, &load.describe(), reserved);
                Hold::Leased(self.queue(request).await?)
            }
        };
        Ok(AdmittedLoad {
            load: load.clone(),
            worker,
            hold,
        })
    }

    /// Queue for `request`, told to the file this wait serves, retiring idle
    /// workers to make room.
    async fn queue(&self, request: WorkerLeaseRequest) -> Result<HostMemoryLease, WorkerError> {
        let observer = super::checkout_wait::MemoryWaitObserver::current();
        self.coordinator
            .queue_for_lease(request, &observer, self.reclaim.as_ref())
            .await
            .map_err(|failure| WorkerError::MemoryGuard(MemoryGuardError::from(failure)))
    }

    /// Validate hosting and physical feasibility. This observation holds no
    /// execution lease and deliberately ignores transient available memory.
    pub(super) fn assess_fa_memory(
        &self,
        memory: &FaRequestMemoryV2,
    ) -> Result<FaHostAssessment, WorkerError> {
        match (self.hosting, memory) {
            (ModelHosting::TestEcho, FaRequestMemoryV2::NoModel) => Ok(FaHostAssessment::ModelFree),
            (
                ModelHosting::Models,
                FaRequestMemoryV2::Projected {
                    workspace, model, ..
                },
            ) => {
                let host = self.coordinator.snapshot().map_err(|failure| {
                    WorkerError::MemoryGuard(MemoryGuardError::LedgerUnusable(failure))
                })?;
                Ok(
                    match FaProjectedDemand::from_projection(*workspace, *model)
                        .assess(FaPhysicalCapacity::from_snapshot(&host))
                    {
                        Ok(feasible) => FaHostAssessment::Fits(feasible),
                        Err(refusal) => FaHostAssessment::Exceeds(refusal),
                    },
                )
            }
            (ModelHosting::Models, FaRequestMemoryV2::NoModel)
            | (ModelHosting::TestEcho, FaRequestMemoryV2::Projected { .. }) => {
                Err(WorkerError::Protocol(
                    "FA preparation memory evidence disagrees with worker hosting".into(),
                ))
            }
        }
    }

    async fn fa_request(
        &self,
        loaded: &Loaded,
        memory: &FaRequestMemoryV2,
    ) -> Result<RequestHold, WorkerError> {
        match self.assess_fa_memory(memory)? {
            FaHostAssessment::ModelFree => Ok(RequestHold::CoveredByLoad),
            FaHostAssessment::Exceeds(refusal) => {
                Err(WorkerError::MemoryGuard(refusal.into_guard_error(format!(
                    "FA model and prepared request on pid {}",
                    loaded.worker
                ))))
            }
            FaHostAssessment::Fits(feasible) => {
                let additional =
                    feasible.additional_reservation(ModelLeaseCoverage::for_loaded(loaded));
                let lease = self
                    .queue(WorkerLeaseRequest::model_request(
                        loaded.worker,
                        &loaded.load.describe(),
                        additional,
                    ))
                    .await?;
                Ok(RequestHold::Additional { _lease: lease })
            }
        }
    }
}

/// What preparing a worker for a load needs: the host's admission of the
/// load's lease, and the stall window the load runs under.
#[derive(Debug, Clone)]
pub struct LoadPreparation {
    /// Admits the load's host-memory lease.
    pub admission: LoadAdmission,
    /// How long the load may go without advancing.
    pub stall: LoadingStall,
}

/// The lease a load holds.
#[derive(Debug)]
enum Hold {
    /// None: the load builds nothing in the worker ([`Residency::Nothing`],
    /// or any load on a test-echo worker).
    Unleased,
    /// The host-memory lease covering the model.
    Leased(HostMemoryLease),
}

/// A load the host admitted, its lease held, before the worker is asked.
///
/// Made only by [`LoadAdmission::admit`]; consumed only by
/// [`AdmittedLoad::confirm`]. Dropped on any other path (a failed or refused
/// load), which releases its lease.
#[derive(Debug)]
pub struct AdmittedLoad {
    load: TaskLoad,
    worker: WorkerPid,
    hold: Hold,
}

impl AdmittedLoad {
    /// The load admitted, for the transport's `ensure_task` request.
    pub(crate) fn load(&self) -> &TaskLoad {
        &self.load
    }

    /// The `AdmittedLoad` to `ConfirmedLoad` transition, once the worker's
    /// `ensure_task` reply named the task: the model is now resident, in the
    /// host's own reading of used memory, so its lease is settled to what it
    /// has yet to use ([`Residency::resident`]), on the blocking pool (the
    /// ledger is a locked file).
    ///
    /// Best effort: a ledger that cannot be rewritten, or a settle task that
    /// never ran, keeps the larger, loading-time reservation, which
    /// over-reserves and never under-reserves.
    pub(crate) async fn confirm(self) -> ConfirmedLoad {
        if let Hold::Leased(lease) = &self.hold {
            let settler = lease.settler();
            let resident = self.load.residency().resident();
            let outcome = crate::blocking::spawn_in_span(move || settler.settle(resident)).await;
            match outcome {
                Ok(Ok(())) => {}
                Ok(Err(error)) => tracing::warn!(
                    %error,
                    "Could not settle a model lease to its resident size; it keeps its loading size"
                ),
                Err(error) => tracing::warn!(
                    %error,
                    "A model lease's settle task did not finish; it keeps its loading size"
                ),
            }
        }
        ConfirmedLoad(self)
    }
}

/// A load its worker confirmed, its lease settled: ready to record. Made only
/// by [`AdmittedLoad::confirm`]; consumed only by [`TaskLoads::record`].
#[derive(Debug)]
pub struct ConfirmedLoad(AdmittedLoad);

/// Proof that one worker completed one [`TaskLoad`], owning the load's
/// host-memory lease for as long as the worker holds the model.
///
/// No public constructor: [`TaskLoads::record`] is the only one.
#[derive(Debug)]
pub struct Loaded {
    load: TaskLoad,
    worker: WorkerPid,
    /// Released when the proof is dropped: when the worker, and so its
    /// recorded loads, goes.
    hold: Hold,
}

/// The minimum reservation still covered by this confirmed load's lease,
/// not measured resident model bytes. Only the loaded-state owner issues it.
#[derive(Debug)]
pub(super) struct ModelLeaseCoverage(MemoryMb);

impl ModelLeaseCoverage {
    fn for_loaded(loaded: &Loaded) -> Self {
        Self(loaded.load.residency().resident())
    }

    pub(super) fn reserved(self) -> MemoryMb {
        self.0
    }

    #[cfg(test)]
    pub(super) fn declared(reserved: MemoryMb) -> Self {
        Self(reserved)
    }
}

impl Loaded {
    /// The load this proves.
    pub fn load(&self) -> &TaskLoad {
        &self.load
    }

    /// Pair an admitted request with its loaded model and take its additional
    /// lease before granting execution. Raw execute packets have no route
    /// through this transition. FA preparation owns the candidate/input
    /// relation; this transition owns the request's host-memory grant.
    async fn run(
        self: &Arc<Self>,
        prepared: ReadyRequest,
        admission: &LoadAdmission,
    ) -> Result<RunPermit, WorkerError> {
        // A batch lease only for a per-batch model the worker really built.
        let hold = match prepared.fa_memory() {
            Some(memory) => admission.fa_request(self, memory).await?,
            None => match (&self.hold, self.load.residency().batch()) {
                (Hold::Unleased, _) | (Hold::Leased(_), None) => RequestHold::CoveredByLoad,
                (Hold::Leased(_), Some(reserved)) => {
                    let request = WorkerLeaseRequest::model_batch(
                        self.worker,
                        &self.load.describe(),
                        reserved,
                    );
                    RequestHold::Additional {
                        _lease: admission.queue(request).await?,
                    }
                }
            },
        };
        Ok(RunPermit {
            loaded: self.clone(),
            prepared,
            _hold: hold,
        })
    }
}

/// The actual transport that performs a load. Implementations are sealed to the
/// three production adapters and the shared declared-effect fixture.
pub(crate) trait TaskLoader: loader_seal::Sealed {
    fn ensure_load(
        &mut self,
        load: &TaskLoad,
        preparation: &LoadPreparation,
    ) -> impl std::future::Future<Output = Result<Arc<Loaded>, WorkerError>> + Send;
}

mod loader_seal {
    pub(crate) trait Sealed {}
    impl Sealed for crate::worker::handle::WorkerHandle {}
    impl Sealed for crate::worker::tcp_handle::TcpWorkerHandle {}
    impl<W> Sealed for &super::super::shared_gpu::channel::SharedGpuChannel<W> {}
}

impl TaskLoader for crate::worker::handle::WorkerHandle {
    async fn ensure_load(
        &mut self,
        load: &TaskLoad,
        preparation: &LoadPreparation,
    ) -> Result<Arc<Loaded>, WorkerError> {
        self.ensure_task(load, preparation).await
    }
}

impl TaskLoader for crate::worker::tcp_handle::TcpWorkerHandle {
    async fn ensure_load(
        &mut self,
        load: &TaskLoad,
        preparation: &LoadPreparation,
    ) -> Result<Arc<Loaded>, WorkerError> {
        self.ensure_task(load, preparation).await
    }
}

impl<W: tokio::io::AsyncWrite + Unpin + Send> TaskLoader
    for &super::shared_gpu::channel::SharedGpuChannel<W>
{
    async fn ensure_load(
        &mut self,
        load: &TaskLoad,
        preparation: &LoadPreparation,
    ) -> Result<Arc<Loaded>, WorkerError> {
        self.ensure_task(load, preparation).await
    }
}

impl LoadPreparation {
    /// Consume preparation on the same adapter that supplies its confirmed load.
    pub(in crate::worker) async fn prepare_on(
        &self,
        prepared: ReadyRequest,
        worker: &mut impl TaskLoader,
    ) -> Result<RunPermit, WorkerError> {
        let loaded = worker.ensure_load(prepared.load(), self).await?;
        loaded.run(prepared, &self.admission).await
    }
}

/// A request ready to run: paired with the proof of its own load and holding
/// any batch lease its model needs. Made only by [`Loaded::run`]; every
/// transport's execute takes one.
#[derive(Debug)]
pub struct RunPermit {
    loaded: Arc<Loaded>,
    prepared: ReadyRequest,
    /// Any additional per-batch or request lease is held until the permit
    /// ends, including cancellation and errors.
    _hold: RequestHold,
}

#[derive(Debug)]
enum RequestHold {
    CoveredByLoad,
    Additional { _lease: HostMemoryLease },
}

impl RunPermit {
    /// The request to send.
    pub fn request(&self) -> &ExecuteRequestV2 {
        self.prepared.request()
    }

    /// Consume execution authority when admitting this invocation's terminal reply.
    pub(in crate::worker) fn finish(
        self,
        reply: Result<crate::types::worker_v2::ExecuteResponseV2, WorkerError>,
    ) -> Result<crate::types::worker_v2::ExecuteResponseV2, WorkerError> {
        use crate::types::worker_v2::{ExecuteOutcomeRef, TaskResultV2};
        let response = reply?;
        let request = self.request();
        if response.request_id() != &request.request_id {
            return Err(WorkerError::Protocol(format!(
                "execute reply {} does not answer request {}",
                response.request_id(),
                request.request_id
            )));
        }
        if let ExecuteOutcomeRef::Success(result) = response.read() {
            use crate::types::worker_v2::InferenceTaskV2 as Task;
            let actual = match result {
                TaskResultV2::FaPreparation(_)
                | TaskResultV2::WhisperTokenTimingResult(_)
                | TaskResultV2::IndexedWordTimingResult(_) => Task::ForcedAlignment,
                TaskResultV2::WhisperChunkResult(_) | TaskResultV2::MonologueAsrResult(_) => {
                    Task::Asr
                }
                TaskResultV2::MorphosyntaxResult(_) => Task::Morphosyntax,
                TaskResultV2::UtsegResult(_) => Task::Utseg,
                TaskResultV2::TranslationResult(_) => Task::Translate,
                TaskResultV2::CorefResult(_) => Task::Coref,
                TaskResultV2::WordSegmentationResult(_) => Task::WordSegmentation,
                TaskResultV2::SpeakerResult(_) => Task::Speaker,
                TaskResultV2::SpeakerEmbeddingResult(_) => Task::SpeakerEmbedding,
                TaskResultV2::OpensmileResult(_) => Task::Opensmile,
                TaskResultV2::AvqiResult(_) => Task::Avqi,
            };
            if actual != request.task {
                return Err(WorkerError::Protocol(format!(
                    "execute reply for {} returned {actual:?}, expected {:?}",
                    request.request_id, request.task
                )));
            }
        }
        Ok(response)
    }

    /// The load it runs on.
    pub fn loaded(&self) -> &Loaded {
        &self.loaded
    }
}

/// One worker's on-demand loads: `Loaded` states by load, absent is
/// `NotLoaded`.
#[derive(Debug, Default)]
pub(crate) struct TaskLoads {
    loaded: HashMap<TaskLoad, Arc<Loaded>>,
}

impl TaskLoads {
    /// The proof for `load`, when this worker already completed it.
    pub(in crate::worker) fn get(&self, load: &TaskLoad) -> Option<Arc<Loaded>> {
        self.loaded.get(load).cloned()
    }

    /// The `ConfirmedLoad` to `Loaded` transition: record a load the worker
    /// confirmed and return its proof, which takes over the settled lease.
    /// The only constructor of [`Loaded`]. A second record of a load already
    /// recorded keeps the first proof and drops the second lease.
    pub(in crate::worker) fn record(&mut self, confirmed: ConfirmedLoad) -> Arc<Loaded> {
        let ConfirmedLoad(AdmittedLoad { load, worker, hold }) = confirmed;
        Arc::clone(
            self.loaded
                .entry(load.clone())
                .or_insert_with(|| Arc::new(Loaded { load, worker, hold })),
        )
    }
}

#[cfg(test)]
pub(crate) mod tests;
