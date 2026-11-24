//! The two shared echo fixtures each own their launch, pipe and load context.
//!
//! Their reactor outlives individual tests. Startup runs on that reactor through
//! publication, so dropping a checkout future cannot leave an untracked spawn.

use std::sync::{Arc, OnceLock};

use tokio::sync::{Mutex, OnceCell, OwnedMutexGuard};

use batchalign::api::{LanguageCode3, WorkerLanguage};
use batchalign::host_memory::NoIdleWorkers;
use batchalign::types::worker_v2::ExecuteResponseV2;
use batchalign::worker::error::WorkerError;
use batchalign::worker::handle::{WorkerConfig, WorkerHandle};
use batchalign::worker::pool::{LoadAdmission, LoadPreparation, ModelHosting, ReadyRequest};
use batchalign::worker::{BatchInferRequest, BatchInferResponse, InferRequest, InferResponse};
use batchalign::worker::{InferTask, LoadingStall, WorkerCapabilities, WorkerHealthResponse};
use batchalign::worker::{WorkerPid, WorkerProfile};

use crate::prerequisite::PythonWorker;

/// Every shared echo scenario owns a fixed launch recipe and one process slot.
#[derive(Clone, Copy)]
pub enum EchoWorker {
    Profile,
    Morphosyntax,
}

struct StartedEcho {
    handle: WorkerHandle,
    load: LoadPreparation,
}

impl EchoWorker {
    fn slot(self) -> &'static OnceCell<Arc<Mutex<StartedEcho>>> {
        static PROFILE: OnceCell<Arc<Mutex<StartedEcho>>> = OnceCell::const_new();
        static MORPHOSYNTAX: OnceCell<Arc<Mutex<StartedEcho>>> = OnceCell::const_new();

        match self {
            Self::Profile => &PROFILE,
            Self::Morphosyntax => &MORPHOSYNTAX,
        }
    }

    async fn start(
        self,
        python: &'static PythonWorker,
    ) -> Result<Arc<Mutex<StartedEcho>>, WorkerError> {
        let config = WorkerConfig {
            profile: WorkerProfile::Stanza,
            task: match self {
                Self::Profile => None,
                Self::Morphosyntax => Some(InferTask::Morphosyntax),
            },
            lang: WorkerLanguage::from(LanguageCode3::eng()),
            test_echo: true,
            ..WorkerConfig::for_tests_with_python(python.interpreter())
        };
        let idle = Arc::new(NoIdleWorkers);
        let load = LoadPreparation {
            admission: LoadAdmission::new(
                config.runtime.host.memory_coordinator(),
                ModelHosting::TestEcho,
                idle.clone(),
            ),
            stall: LoadingStall::DEFAULT_ENSURE_TASK,
        };
        let handle = WorkerHandle::spawn(config, idle).await?;
        Ok(Arc::new(Mutex::new(StartedEcho { handle, load })))
    }

    pub async fn checkout(self) -> Result<EchoLease, WorkerError> {
        let slot = self.slot();
        let worker = match slot.get() {
            Some(existing) => Arc::clone(existing),
            None => {
                let python = PythonWorker::require();
                // The shared runtime owns get-or-init, spawn and publication.
                // Cancellation of an awaiting test only drops its join handle.
                worker_runtime()
                    .spawn(async move {
                        slot.get_or_try_init(|| self.start(python))
                            .await
                            .map(Arc::clone)
                    })
                    .await
                    .map_err(|error| WorkerError::Io(std::io::Error::other(error)))??
            }
        };
        Ok(EchoLease(worker.lock_owned().await))
    }
}

/// Exclusive access to the admitted transport and its own load preparation.
pub struct EchoLease(OwnedMutexGuard<StartedEcho>);

impl EchoLease {
    pub fn pid(&self) -> WorkerPid {
        self.0.handle.pid()
    }

    pub fn runtime_identity(&self) -> &batchalign::worker::runtime_identity::WorkerRuntimeIdentity {
        self.0.handle.runtime_identity()
    }

    pub async fn health_check(&mut self) -> Result<WorkerHealthResponse, WorkerError> {
        self.0.handle.health_check().await
    }

    pub async fn capabilities(&mut self) -> Result<WorkerCapabilities, WorkerError> {
        self.0.handle.capabilities().await
    }

    pub async fn infer(&mut self, request: &InferRequest) -> Result<InferResponse, WorkerError> {
        self.0.handle.infer(request).await
    }

    pub async fn batch_infer(
        &mut self,
        request: &BatchInferRequest,
    ) -> Result<BatchInferResponse, WorkerError> {
        self.0.handle.batch_infer(request).await
    }

    pub async fn execute_v2(
        &mut self,
        request: ReadyRequest,
    ) -> Result<ExecuteResponseV2, WorkerError> {
        let StartedEcho { handle, load } = &mut *self.0;
        handle.execute_v2(request, load).await
    }
}

fn worker_runtime() -> &'static tokio::runtime::Handle {
    static RUNTIME: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    RUNTIME
        .get_or_init(|| {
            tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .enable_all()
                .thread_name("test-worker-pool")
                .build()
                .expect("build the shared test-worker runtime")
        })
        .handle()
}
