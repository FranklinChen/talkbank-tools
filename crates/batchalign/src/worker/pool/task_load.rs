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
//! Loaded --Loaded::run (a per-batch model's batch lease; may queue)-------> RunPermit
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
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TaskLoad {
    task: InferTask,
    /// The task's own keys in the worker's dispatch spelling; empty when the
    /// task's load reads none.
    overrides: BTreeMap<String, String>,
    /// What the load keeps resident, from the same typed keys the overrides
    /// were written from: a function of the two fields above, so it never
    /// makes two loads differ that they do not.
    residency: Residency,
}

impl TaskLoad {
    /// The load `task` needs, given the engines `selected` names.
    pub(super) fn new(task: InferTask, selected: EngineOverrides) -> Self {
        let keys = EngineSelection::task_keys(task, selected);
        Self {
            task,
            residency: Residency::of(task, &keys),
            overrides: keys.dispatch_overrides(),
        }
    }

    /// The load a request needs: its pool task and the engines it names.
    pub fn for_request(request: &ExecuteRequestV2) -> Self {
        let task = infer_task_for_execute_v2(request.task);
        Self::new(task, execute_request_overrides(request))
    }

    /// The task loaded.
    pub fn task(&self) -> InferTask {
        self.task
    }

    /// What the load keeps resident in its worker.
    pub fn residency(&self) -> Residency {
        self.residency
    }

    /// The overrides the `ensure_task` request carries, `None` when empty.
    pub(crate) fn wire_overrides(&self) -> Option<&BTreeMap<String, String>> {
        (!self.overrides.is_empty()).then_some(&self.overrides)
    }

    /// The load as a ledger label names it: the task and its engines.
    fn describe(&self) -> String {
        let mut what = self.task.wire_name().to_owned();
        for (key, value) in &self.overrides {
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
        let hold = match (self.hosting, load.residency.loading()) {
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
            let resident = self.load.residency.resident();
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
        Self(loaded.load.residency.resident())
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

    /// Refuse to run `request` on this proof unless it is the request's own
    /// load: the one place a proof and a request are paired.
    fn admit(&self, request: &ExecuteRequestV2) -> Result<(), WorkerError> {
        let needed = TaskLoad::for_request(request);
        if needed == self.load {
            Ok(())
        } else {
            Err(WorkerError::Protocol(format!(
                "request {} needs {needed:?}, but the worker was prepared for {:?}",
                request.request_id, self.load
            )))
        }
    }

    /// Pair an admitted request with its loaded model and take its additional
    /// lease before granting execution. Raw execute packets have no route
    /// through this transition. FA preparation owns the candidate/input
    /// relation; this transition owns the request's host-memory grant.
    pub async fn run<'a>(
        &'a self,
        prepared: ReadyRequest,
        admission: &LoadAdmission,
    ) -> Result<RunPermit<'a>, WorkerError> {
        let request = prepared.request();
        self.admit(request)?;
        // A batch lease only for a per-batch model the worker really built.
        let hold = match prepared.fa_memory() {
            Some(memory) => admission.fa_request(self, memory).await?,
            None => match (&self.hold, self.load.residency.batch()) {
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
            loaded: self,
            prepared,
            _hold: hold,
        })
    }
}

/// A request ready to run: paired with the proof of its own load and holding
/// any batch lease its model needs. Made only by [`Loaded::run`]; every
/// transport's execute takes one.
#[derive(Debug)]
pub struct RunPermit<'a> {
    loaded: &'a Loaded,
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

impl<'a> RunPermit<'a> {
    /// The request to send.
    pub fn request(&self) -> &ExecuteRequestV2 {
        self.prepared.request()
    }

    /// The load it runs on.
    pub fn loaded(&self) -> &'a Loaded {
        self.loaded
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
pub(crate) mod tests {
    use super::*;
    use crate::api::MemoryMb;
    use crate::types::engines::SpeakerEngineName;
    use crate::types::worker_v2::SpeakerEmbeddingBackendV2;

    /// An admission against a fresh ledger on the declared test host (14,000
    /// MB available, 8,000 MB reserve), and the directory holding it.
    pub(crate) fn test_admission() -> (tempfile::TempDir, LoadAdmission) {
        let dir = tempfile::tempdir().expect("ledger dir");
        let host =
            crate::host_facts::HostSnapshot::for_tests(&crate::config::ServerConfig::default())
                .runtime_host()
                .with_ledger(crate::host_memory::HostMemoryRuntimeConfig::from_sources(
                    dir.path().join("ledger.json"),
                    MemoryMb(8_000),
                    crate::config::WorkerStartupLimit::literal::<1>(),
                ));
        (
            dir,
            LoadAdmission::new(
                host.memory_coordinator(),
                ModelHosting::Models,
                Arc::new(crate::host_memory::NoIdleWorkers),
            ),
        )
    }

    fn reserved(admission: &LoadAdmission) -> MemoryMb {
        admission
            .coordinator
            .snapshot()
            .expect("snapshot")
            .active_reserved_mb
    }

    pub(crate) fn ready(request: &ExecuteRequestV2) -> ReadyRequest {
        match super::super::fa_preparation::DispatchPreparation::begin(request)
            .expect("ordinary request")
        {
            super::super::fa_preparation::DispatchPreparation::Ready(ready) => ready,
            super::super::fa_preparation::DispatchPreparation::ForcedAlignment(_) => {
                panic!("FA requires its preparation reply")
            }
        }
    }

    fn this_worker() -> WorkerPid {
        WorkerPid(std::process::id())
    }

    fn speaker(selected: EngineOverrides) -> TaskLoad {
        TaskLoad::new(InferTask::Speaker, selected)
    }

    fn local_pyannote() -> TaskLoad {
        speaker(EngineOverrides {
            speaker: Some(SpeakerEngineName::Pyannote),
            ..EngineOverrides::default()
        })
    }

    /// Two speaker models on one worker are two loads; a repeated record is
    /// the same proof and takes no second lease.
    #[tokio::test]
    async fn a_load_is_its_task_and_model() {
        let (_dir, admission) = test_admission();
        let local = local_pyannote();
        let embedding = speaker(EngineOverrides {
            speaker_embedding: Some(SpeakerEmbeddingBackendV2::Pyannote),
            ..EngineOverrides::default()
        });
        assert_ne!(local, embedding);

        let mut loads = TaskLoads::default();
        assert!(loads.get(&local).is_none());
        let first = loads.record(
            admission
                .admit(&local, this_worker())
                .await
                .expect("fits")
                .confirm()
                .await,
        );
        let again = loads.record(
            admission
                .admit(&local, this_worker())
                .await
                .expect("fits")
                .confirm()
                .await,
        );
        assert!(Arc::ptr_eq(&first, &again));
        assert!(loads.get(&embedding).is_none());
        assert!(Arc::ptr_eq(&first, &loads.get(&local).expect("recorded")));
        assert_eq!(
            reserved(&admission),
            local.residency().resident(),
            "one resident lease, settled to its margin; the repeat's lease is gone"
        );
    }

    /// The order, end to end: the lease is taken BEFORE the worker is asked
    /// (counted in full while the model loads), the proof takes it over and
    /// settles it, and it is released when the proof is dropped.
    #[tokio::test]
    async fn a_proof_owns_its_load_lease() {
        let (_dir, admission) = test_admission();
        let load = local_pyannote();
        let residency = load.residency();

        let admitted = admission.admit(&load, this_worker()).await.expect("fits");
        assert_eq!(
            Some(reserved(&admission)),
            residency.loading(),
            "the lease is held before the worker is asked to load"
        );

        let mut loads = TaskLoads::default();
        let loaded = loads.record(admitted.confirm().await);
        assert_eq!(reserved(&admission), residency.resident());

        drop(loads);
        drop(loaded);
        assert_eq!(reserved(&admission), MemoryMb(0), "released with the proof");
    }

    /// A load that fails after its lease was taken drops the admitted load,
    /// and the lease with it; no `Loaded` exists.
    #[tokio::test]
    async fn a_failed_load_releases_its_lease() {
        let (_dir, admission) = test_admission();
        let load = local_pyannote();
        let mut loads = TaskLoads::default();

        async fn failing_round_trip(_load: &TaskLoad) -> Result<(), WorkerError> {
            Err(WorkerError::Protocol("the worker refused the load".into()))
        }
        let outcome: Result<Arc<Loaded>, WorkerError> = async {
            let admitted = admission.admit(&load, this_worker()).await?;
            failing_round_trip(admitted.load()).await?;
            Ok(loads.record(admitted.confirm().await))
        }
        .await;

        assert!(outcome.is_err());
        assert!(loads.get(&load).is_none(), "no proof for a failed load");
        assert_eq!(reserved(&admission), MemoryMb(0), "no lease outlives it");
    }

    /// A load that builds nothing, and any load on a test-echo worker, takes
    /// no lease.
    #[tokio::test]
    async fn a_load_with_nothing_resident_takes_no_lease() {
        let (_dir, admission) = test_admission();
        let hosted = speaker(EngineOverrides::default());
        assert_eq!(hosted.residency(), Residency::Nothing);
        let mut loads = TaskLoads::default();
        let _hosted = loads.record(
            admission
                .admit(&hosted, this_worker())
                .await
                .expect("no lease")
                .confirm()
                .await,
        );
        let echo = LoadAdmission::new(
            admission.coordinator.clone(),
            ModelHosting::TestEcho,
            Arc::new(crate::host_memory::NoIdleWorkers),
        );
        let _echo = loads.record(
            echo.admit(&local_pyannote(), this_worker())
                .await
                .expect("echo")
                .confirm()
                .await,
        );
        assert_eq!(reserved(&admission), MemoryMb(0));
    }

    fn speaker_request(backend: crate::types::worker_v2::SpeakerBackendV2) -> ExecuteRequestV2 {
        use crate::types::worker_v2::{
            InferenceTaskV2, SpeakerInputV2, SpeakerPreparedAudioInputV2, SpeakerRequestV2,
            TaskRequestV2, WorkerArtifactIdV2, WorkerRequestIdV2,
        };
        ExecuteRequestV2 {
            request_id: WorkerRequestIdV2::from("task-load-test"),
            task: InferenceTaskV2::Speaker,
            payload: TaskRequestV2::Speaker(SpeakerRequestV2 {
                backend,
                input: SpeakerInputV2::PreparedAudio(SpeakerPreparedAudioInputV2 {
                    audio_ref_id: WorkerArtifactIdV2::from("audio-1"),
                }),
                expected_speakers: None,
            }),
            attachments: Vec::new(),
        }
    }

    /// A proof runs only the request it was prepared for: a worker prepared
    /// for pyannoteAI (nothing loaded) cannot run a local-pyannote request.
    #[tokio::test]
    async fn a_proof_admits_only_its_own_load() {
        use crate::types::worker_v2::SpeakerBackendV2;
        let (_dir, admission) = test_admission();
        let hosted = speaker_request(SpeakerBackendV2::PyannoteAi);
        let local = speaker_request(SpeakerBackendV2::Pyannote);
        let loaded = TaskLoads::default().record(
            admission
                .admit(&TaskLoad::for_request(&hosted), this_worker())
                .await
                .expect("no lease")
                .confirm()
                .await,
        );
        assert!(loaded.run(ready(&hosted), &admission).await.is_ok());
        assert!(matches!(
            loaded.run(ready(&local), &admission).await,
            Err(WorkerError::Protocol(_))
        ));
    }

    /// A per-batch model (coreference) reserves nothing once loaded, and each
    /// request holds its batch's own lease for as long as the permit lives.
    #[tokio::test]
    async fn a_per_batch_model_leases_each_batch() {
        let (_dir, admission) = test_admission();
        let request = coref_request();
        let load = TaskLoad::for_request(&request);
        let residency = load.residency();
        let loaded = TaskLoads::default().record(
            admission
                .admit(&load, this_worker())
                .await
                .expect("fits")
                .confirm()
                .await,
        );
        assert_eq!(reserved(&admission), MemoryMb(0));

        let permit = loaded
            .run(ready(&request), &admission)
            .await
            .expect("the batch fits");
        assert_eq!(Some(reserved(&admission)), residency.batch());
        drop(permit);
        assert_eq!(reserved(&admission), MemoryMb(0), "released with the batch");
    }

    fn projected_fa_request(workspace_mb: u64) -> ReadyRequest {
        use crate::types::worker_v2::{FaAllocationBasisV2, FaModelBytesV2, FaWorkspaceBytesV2};
        super::super::fa_preparation::tests::prepared_request(FaRequestMemoryV2::Projected {
            workspace: FaWorkspaceBytesV2::try_from(workspace_mb * 1024 * 1024)
                .expect("positive workspace"),
            model: FaModelBytesV2::try_from(1024 * 1024).expect("declared model storage"),
            basis: FaAllocationBasisV2::MmsCpu {
                torch_version: "declared-test-encoder".into(),
                torchaudio_revision: "declared-test-layout".into(),
            },
        })
    }

    #[tokio::test]
    async fn fa_execution_owns_its_additional_lease_and_releases_it_on_drop() {
        let (_dir, admission) = test_admission();
        let prepared = projected_fa_request(64);
        let load = TaskLoad::for_request(prepared.request());
        let loaded = TaskLoads::default().record(
            admission
                .admit(&load, this_worker())
                .await
                .expect("load fits")
                .confirm()
                .await,
        );
        let resident = reserved(&admission);
        let permit = loaded
            .run(prepared, &admission)
            .await
            .expect("request fits");
        assert_eq!(
            reserved(&admission).0 - resident.0,
            80,
            "64 MiB workspace plus the existing 25 percent estimate"
        );
        drop(permit);
        assert_eq!(
            reserved(&admission),
            resident,
            "the request lease cannot outlive its permit"
        );
    }

    #[tokio::test]
    async fn a_real_model_cannot_run_with_a_model_free_preparation() {
        let (_dir, admission) = test_admission();
        let prepared =
            super::super::fa_preparation::tests::prepared_request(FaRequestMemoryV2::NoModel);
        let load = TaskLoad::for_request(prepared.request());
        let loaded = TaskLoads::default().record(
            admission
                .admit(&load, this_worker())
                .await
                .expect("load fits")
                .confirm()
                .await,
        );
        assert!(matches!(
            loaded.run(prepared, &admission).await,
            Err(WorkerError::Protocol(_))
        ));
    }

    #[tokio::test]
    async fn a_prepared_fa_request_that_cannot_fit_the_host_never_queues() {
        let (_dir, admission) = test_admission();
        let prepared = projected_fa_request(1_000_000);
        let load = TaskLoad::for_request(prepared.request());
        let loaded = TaskLoads::default().record(
            admission
                .admit(&load, this_worker())
                .await
                .expect("load fits")
                .confirm()
                .await,
        );
        assert!(matches!(
            loaded.run(prepared, &admission).await,
            Err(WorkerError::MemoryGuard(
                MemoryGuardError::ExceedsHost { .. }
            ))
        ));
    }

    fn coref_request() -> ExecuteRequestV2 {
        use crate::types::worker_v2::{
            CorefRequestV2, InferenceTaskV2, TaskRequestV2, WorkerArtifactIdV2, WorkerRequestIdV2,
        };
        ExecuteRequestV2 {
            request_id: WorkerRequestIdV2::from("coref-test"),
            task: InferenceTaskV2::Coref,
            payload: TaskRequestV2::Coref(CorefRequestV2 {
                lang: crate::api::LanguageCode3::eng(),
                payload_ref_id: WorkerArtifactIdV2::from("payload"),
                item_count: 1,
            }),
            attachments: Vec::new(),
        }
    }

    /// A task's load reads only its own keys, so an FA load is the same
    /// whatever else the worker's selection names.
    #[test]
    fn a_load_keeps_only_its_tasks_keys() {
        let named = EngineOverrides {
            fa: Some(crate::types::engines::FaEngineName::Wave2Vec),
            asr: Some(crate::types::engines::AsrEngineName::Whisper),
            ..EngineOverrides::default()
        };
        let fa_only = EngineOverrides {
            fa: Some(crate::types::engines::FaEngineName::Wave2Vec),
            ..EngineOverrides::default()
        };
        assert_eq!(
            TaskLoad::new(InferTask::Fa, named),
            TaskLoad::new(InferTask::Fa, fa_only)
        );
        assert_eq!(
            TaskLoad::new(InferTask::Coref, EngineOverrides::default()).wire_overrides(),
            None
        );
    }
}
