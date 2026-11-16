use super::*;
use crate::api::MemoryMb;
use crate::types::engines::SpeakerEngineName;
use crate::types::worker_v2::SpeakerEmbeddingBackendV2;

/// An admission against a fresh ledger on the declared test host (14,000
/// MB available, 8,000 MB reserve), and the directory holding it.
pub(crate) fn test_admission() -> (tempfile::TempDir, LoadAdmission) {
    let dir = tempfile::tempdir().expect("ledger dir");
    let host = crate::host_facts::HostSnapshot::for_tests(&crate::config::ServerConfig::default())
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

/// A declared load-effect adapter for contracts that exercise real leases
/// without starting a model. Preparation derives its load from the carried
/// request, then confirms the declared effect through the normal cache.
struct DeclaredLoader {
    loads: TaskLoads,
    worker: WorkerPid,
}
impl loader_seal::Sealed for DeclaredLoader {}
impl TaskLoader for DeclaredLoader {
    async fn ensure_load(
        &mut self,
        load: &TaskLoad,
        preparation: &LoadPreparation,
    ) -> Result<Arc<Loaded>, WorkerError> {
        if let Some(loaded) = self.loads.get(load) {
            return Ok(loaded);
        }
        let admitted = preparation.admission.admit(load, self.worker).await?;
        Ok(self.loads.record(admitted.confirm().await))
    }
}
pub(crate) async fn declared_invocation(
    prepared: ReadyRequest,
    admission: &LoadAdmission,
) -> RunPermit {
    LoadPreparation {
        admission: admission.clone(),
        stall: LoadingStall::DEFAULT_ENSURE_TASK,
    }
    .prepare_on(
        prepared,
        &mut DeclaredLoader {
            loads: TaskLoads::default(),
            worker: this_worker(),
        },
    )
    .await
    .expect("declared invocation fits")
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

/// Guards: actual ledger reservations follow loading, settled cache reuse,
/// load failure, hosted absence and echo hosting through the same load producers.
/// Kind: contract
/// Not a type: the ledger persists measured reservation effects, and residency
/// policy chooses distinct loading, resident and model-free amounts.
/// State: an isolated real ledger observes lease creation, settlement and Drop.
/// Concurrency: every lease uses this process PID and a test-owned directory.
#[tokio::test]
async fn load_effects_preserve_settlement_reuse_failure_and_model_absence() {
    let (_dir, admission) = test_admission();
    let local = local_pyannote();
    let admitted = admission.admit(&local, this_worker()).await.unwrap();
    assert_eq!(Some(reserved(&admission)), local.residency().loading());
    let mut loads = TaskLoads::default();
    let first = loads.record(admitted.confirm().await);
    assert_eq!(reserved(&admission), local.residency().resident());
    let repeated = loads.record(
        admission
            .admit(&local, this_worker())
            .await
            .unwrap()
            .confirm()
            .await,
    );
    assert!(Arc::ptr_eq(&first, &repeated));
    assert!(Arc::ptr_eq(&first, &loads.get(&local).unwrap()));
    assert_eq!(reserved(&admission), local.residency().resident());
    drop((loads, first, repeated));
    assert_eq!(reserved(&admission), MemoryMb(0));

    // A declined external load never advances its admitted lease to confirmation.
    let failed = admission.admit(&local, this_worker()).await.unwrap();
    assert_eq!(Some(reserved(&admission)), local.residency().loading());
    drop(failed);
    assert_eq!(reserved(&admission), MemoryMb(0));
    let hosted = speaker(EngineOverrides::default());
    let mut declared = DeclaredLoader {
        loads: TaskLoads::default(),
        worker: this_worker(),
    };
    let normal = LoadPreparation {
        admission: admission.clone(),
        stall: LoadingStall::DEFAULT_ENSURE_TASK,
    };
    let hosted_proof = declared.ensure_load(&hosted, &normal).await.unwrap();
    let echo = LoadPreparation {
        admission: LoadAdmission::new(
            admission.coordinator.clone(),
            ModelHosting::TestEcho,
            Arc::new(crate::host_memory::NoIdleWorkers),
        ),
        stall: LoadingStall::DEFAULT_ENSURE_TASK,
    };
    let echo_proof = declared.ensure_load(&local, &echo).await.unwrap();
    assert_eq!(reserved(&admission), MemoryMb(0));
    drop((declared, hosted_proof, echo_proof));
}

/// Guards: a coreference batch owns its real lease after the load producer ends,
/// and final request ownership releases that batch reservation.
/// Kind: contract
/// Not a type: the filesystem ledger's observable sum and release amounts follow
/// the residency policy and actual lease effects beyond the consuming API.
/// State: a test-owned ledger records the batch through the sealed declared loader.
/// Concurrency: the producer and request share only their isolated admission coordinator.
#[tokio::test]
async fn a_per_batch_model_leases_each_batch() {
    let (_dir, admission) = test_admission();
    let request = coref_request();
    let expected = ready(&request).load().residency().batch().unwrap();
    let preparation = LoadPreparation {
        admission: admission.clone(),
        stall: LoadingStall::DEFAULT_ENSURE_TASK,
    };
    let mut declared = DeclaredLoader {
        loads: TaskLoads::default(),
        worker: this_worker(),
    };
    let first = preparation
        .prepare_on(ready(&request), &mut declared)
        .await
        .unwrap();
    assert_eq!(reserved(&admission), expected);
    drop(declared);
    assert_eq!(reserved(&admission), expected);
    drop(first);
    assert_eq!(reserved(&admission), MemoryMb(0));
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

/// Guards: FA grants its projected workspace once and refuses model-free or
/// physically impossible reports before taking an execution lease.
/// Kind: contract
/// Not a type: the worker's memory report and current host capacity are external
/// facts, while the allocation margin is a deliberate admission policy.
/// State: an isolated real ledger observes the model and request reservations.
/// Concurrency: one declared worker cache and host coordinator belong to this runtime.
#[tokio::test]
async fn fa_reports_admit_workspace_or_preserve_the_original_refusal() {
    enum Report {
        Fits,
        ModelFreeMismatch,
        ExceedsHost,
    }
    let (_dir, admission) = test_admission();
    let preparation = LoadPreparation {
        admission: admission.clone(),
        stall: LoadingStall::DEFAULT_ENSURE_TASK,
    };
    let mut declared = DeclaredLoader {
        loads: TaskLoads::default(),
        worker: this_worker(),
    };
    for report in [Report::Fits, Report::ModelFreeMismatch, Report::ExceedsHost] {
        let ready = match report {
            Report::Fits => projected_fa_request(64),
            Report::ModelFreeMismatch => {
                super::super::fa_preparation::tests::prepared_request(FaRequestMemoryV2::NoModel)
            }
            Report::ExceedsHost => projected_fa_request(1_000_000),
        };
        // Observe the model effect separately. The request still enters only
        // through the sealed actual producer and reuses its confirmed cache.
        let loaded = declared
            .ensure_load(ready.load(), &preparation)
            .await
            .unwrap();
        let resident = reserved(&admission);
        match report {
            Report::Fits => {
                let permit = preparation.prepare_on(ready, &mut declared).await.unwrap();
                assert_eq!(
                    reserved(&admission).0 - resident.0,
                    80,
                    "64 MiB plus the allocation margin"
                );
                drop(permit);
            }
            Report::ModelFreeMismatch => assert!(matches!(
                preparation.prepare_on(ready, &mut declared).await,
                Err(WorkerError::Protocol(_))
            )),
            Report::ExceedsHost => assert!(matches!(
                preparation.prepare_on(ready, &mut declared).await,
                Err(WorkerError::MemoryGuard(
                    MemoryGuardError::ExceedsHost { .. }
                ))
            )),
        }
        assert_eq!(reserved(&admission), resident);
        drop(loaded);
    }
    drop(declared);
    assert_eq!(reserved(&admission), MemoryMb(0));
}

pub(crate) fn coref_request() -> ExecuteRequestV2 {
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

/// Guards: FA ignores foreign ASR keys, coreference has no dispatch overrides,
/// and local diarization and speaker embedding remain different load identities.
/// Kind: policy
/// Not a type: assigning engine keys to tasks is a semantic choice whose selected
/// values can change while all enum variants and constructors still compile.
/// State: none
/// Concurrency: none
#[test]
fn a_load_keeps_only_its_tasks_keys() {
    assert_ne!(
        local_pyannote(),
        speaker(EngineOverrides {
            speaker_embedding: Some(SpeakerEmbeddingBackendV2::Pyannote),
            ..EngineOverrides::default()
        })
    );
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
