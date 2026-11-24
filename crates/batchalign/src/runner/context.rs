//! Type definitions for the runner module's execution context, host bundles,
//! and orchestration traits.
//!
//! These are the shared data structures that thread through job dispatch:
//! - `RunnerExecutionContext`: worker pool + cache + engine metadata
//! - `ExecutionEngine`: thin wrapper that dispatches to the routing layer
//! - `DispatchHostContext`: read-only server/config context for dispatch
//! - `ServerExecutionHost` / `DirectExecutionHost`, host-owned bundles
//! - `QueuedJobOrchestrator`: host-owned re-queue policy trait

use std::sync::Arc;

use async_trait::async_trait;

use crate::api::NumWorkers;
use crate::cache::UtteranceCache;
use crate::config::ServerConfig;
use crate::runner::util::{RunnerEventSink, StoreRunnerEventSink};
use crate::store::{AdmittedRun, JobStore, RunnerJobSnapshot};
use crate::worker::pool::WorkerPool;

/// Shared dependencies needed to build per-job runner tasks.
///
/// Carries no capability view: dispatch reads the selected worker's admitted
/// report from the pool for each job, and test-echo workers answer every task.
#[derive(Clone)]
pub struct RunnerExecutionContext {
    pub(super) pool: Arc<WorkerPool>,
    pub(super) cache: Arc<UtteranceCache>,
    pub(super) test_echo_mode: bool,
}

impl RunnerExecutionContext {
    pub(crate) const fn new(
        pool: Arc<WorkerPool>,
        cache: Arc<UtteranceCache>,
        test_echo_mode: bool,
    ) -> Self {
        Self {
            pool,
            cache,
            test_echo_mode,
        }
    }
}

/// Shared execution engine built on one resolved runtime context.
#[derive(Clone)]
pub struct ExecutionEngine {
    pub(super) context: RunnerExecutionContext,
}

impl ExecutionEngine {
    /// Build one shared execution engine over a resolved runtime context.
    pub(crate) const fn new(context: RunnerExecutionContext) -> Self {
        Self { context }
    }

    pub(super) async fn dispatch_job(
        &self,
        request: JobDispatchRequest,
    ) -> Result<DispatchEnd, crate::error::ServerError> {
        super::routing::dispatch_job_with_execution_context(request, &self.context).await
    }

    /// Borrow the underlying worker pool for cancel-driven shutdown.
    /// The cancel pathway in `ServerBackend::cancel_job` detaches the job's
    /// in-flight workers from the pool before publishing the cancel and
    /// terminates them after; without this accessor, the backend has no way
    /// to reach the pool.
    pub(crate) const fn pool(&self) -> &Arc<WorkerPool> {
        &self.context.pool
    }
}

/// Read-only host/runtime context consulted during shared command dispatch.
///
/// This keeps performance policy and media-resolution config explicit host
/// concerns without threading the full mutable `JobStore` through command code.
///
/// Holds an [`EffectiveConfig`] resolved once at construction from the host's
/// `ServerConfig` plus the live `HostFacts` snapshot. Per-job dispatch reads
/// from this resolved view rather than re-detecting host facts on every call,
/// which is the architectural seam the host-facts migration introduced (see
/// `book/src/batchalign/developer/host-facts.md`).
#[derive(Clone)]
pub struct DispatchHostContext {
    trace_store: Arc<crate::trace_store::TraceStore>,
    config: Arc<ServerConfig>,
    /// The store's host snapshot: job sizing and the dispatch plans take the
    /// tier, policy and resolved knobs from it rather than reading the machine.
    host: Arc<crate::host_facts::HostSnapshot>,
    sink: Arc<dyn RunnerEventSink>,
}

impl DispatchHostContext {
    /// The dispatch context of the run that `claim` owns: its sink writes
    /// only as that run. The only production route, so dispatch cannot hold
    /// a sink that is not bound to a run.
    pub(crate) fn for_run(claim: &AdmittedRun) -> Self {
        Self::with_sink(
            StoreRunnerEventSink::for_run(claim),
            Arc::clone(claim.store()),
        )
    }

    /// A policy fixture uses the shared effect recorder and one declared host.
    #[cfg(test)]
    pub(crate) fn for_test_effects() -> Self {
        Self::for_recorded_effects(Arc::new(super::util::test_sink::RecordingSink::default()))
    }

    #[cfg(test)]
    pub(crate) fn for_recorded_effects(sink: Arc<super::util::test_sink::RecordingSink>) -> Self {
        let config = Arc::new(ServerConfig::default());
        let host = Arc::new(crate::host_facts::HostSnapshot::for_tests(&config));
        Self {
            config,
            host,
            trace_store: Arc::new(crate::trace_store::TraceStore::new()),
            sink,
        }
    }

    fn with_sink(sink: Arc<dyn RunnerEventSink>, store: Arc<JobStore>) -> Self {
        // The host was read once, where the server was composed; per-job
        // dispatch reads that snapshot rather than re-detecting facts. Runtime
        // memory pressure is the live memory guard's (`worker::memory_guard`).
        Self {
            config: Arc::new(store.config().clone()),
            host: store.host().clone(),
            sink,
            trace_store: Arc::clone(store.trace_store_handle()),
        }
    }

    pub(in crate::runner) fn host_snapshot(&self) -> &crate::host_facts::HostSnapshot {
        &self.host
    }

    /// The host-memory ledger coordinator, deciding on the store's host:
    /// the live machine in a server, the declared one in a test.
    pub(crate) fn memory_coordinator(&self) -> crate::host_memory::HostMemoryCoordinator {
        self.host.memory_coordinator()
    }

    /// The host's execution policy, from the store's snapshot.
    pub(crate) fn host_policy(&self) -> crate::host_policy::HostExecutionPolicy {
        self.host.policy()
    }

    pub(crate) fn config(&self) -> &ServerConfig {
        self.config.as_ref()
    }

    pub(crate) fn sink(&self) -> &Arc<dyn RunnerEventSink> {
        &self.sink
    }

    pub(crate) fn media_mapping_root(
        &self,
        key: &str,
    ) -> Option<&batchalign_types::paths::ServerPath> {
        self.config
            .media_mappings
            .get(&batchalign_types::paths::MediaMappingKey::new(key))
    }

    pub(crate) fn media_roots(&self) -> &[batchalign_types::paths::ServerPath] {
        &self.config.media_roots
    }

    pub(crate) fn trace_store(&self) -> &crate::trace_store::TraceStore {
        self.trace_store.as_ref()
    }
}

/// The immutable snapshot and run-bound host context derived from one claim.
/// Consumers cannot independently assemble their source and event sink.
pub(super) struct ClaimedDispatch {
    job: Arc<RunnerJobSnapshot>,
    host: DispatchHostContext,
}

pub(super) struct ClaimedDispatchParts {
    pub(super) job: Arc<RunnerJobSnapshot>,
    pub(super) host: DispatchHostContext,
}

impl ClaimedDispatch {
    pub(super) fn for_run(claim: &AdmittedRun) -> Self {
        Self {
            job: Arc::clone(claim.snapshot()),
            host: DispatchHostContext::for_run(claim),
        }
    }

    pub(super) const fn job(&self) -> &Arc<RunnerJobSnapshot> {
        &self.job
    }
    pub(super) const fn host(&self) -> &DispatchHostContext {
        &self.host
    }

    pub(super) fn into_parts(self) -> ClaimedDispatchParts {
        ClaimedDispatchParts {
            job: self.job,
            host: self.host,
        }
    }
}

/// Shared server-owned host dependencies needed to build per-job runner tasks.
#[derive(Clone)]
pub struct ServerExecutionHost {
    pub(super) store: Arc<JobStore>,
    pub(super) engine: ExecutionEngine,
    pub(super) orchestrator: Arc<dyn QueuedJobOrchestrator>,
}

impl ServerExecutionHost {
    /// Build the server-owned host bundle around one execution engine.
    pub(crate) fn new(
        store: Arc<JobStore>,
        engine: ExecutionEngine,
        orchestrator: Arc<dyn QueuedJobOrchestrator>,
    ) -> Self {
        Self {
            store,
            engine,
            orchestrator,
        }
    }

    /// Access the underlying worker pool. Backends call this from
    /// `cancel_job` to detach and then terminate every in-flight worker
    /// registered to the cancelled job (see `worker/pool/job_tracker.rs`).
    pub(crate) const fn pool(&self) -> &Arc<WorkerPool> {
        self.engine.pool()
    }
}

/// Shared direct-execution host dependencies needed to run one inline job.
#[derive(Clone)]
pub struct DirectExecutionHost {
    pub(super) store: Arc<JobStore>,
    pub(super) engine: ExecutionEngine,
}

impl DirectExecutionHost {
    /// Build one direct-execution host bundle around one execution engine.
    pub(crate) const fn new(store: Arc<JobStore>, engine: ExecutionEngine) -> Self {
        Self { store, engine }
    }
}

/// Execution-phase request handed from the host-owned runner wrapper into the
/// shared dispatch kernel.
pub(super) struct JobDispatchRequest {
    pub(super) preflight: super::preflight::PreflightedJob,
    pub(super) num_workers: NumWorkers,
}

// Reason: re-exports an item that is visible in this crate only; Clippy's
// suggested `pub` does not compile (E0365).
#[allow(clippy::redundant_pub_crate)]
pub(crate) use crate::store::MemoryRequeueOutcome as MemoryGateRejectionDisposition;

/// A whole job that dispatch declined to run, with the reason the run's
/// failure records on the job and on every file it leaves unfinished.
///
/// A value, not a store write: dispatch hands it back to the consuming run
/// scope, which alone holds the claim that may fail the run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JobRefusal(String);

impl JobRefusal {
    /// Only the runner's routing refuses a whole job.
    pub(super) fn new(reason: impl Into<String>) -> Self {
        Self(reason.into())
    }

    pub(crate) fn into_reason(self) -> String {
        self.0
    }
}

impl std::fmt::Display for JobRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// How command dispatch ended for a claimed run.
#[derive(Debug, Clone, PartialEq, Eq)]
#[must_use = "a refusal must reach the run scope that owns the claim"]
pub enum DispatchEnd {
    /// Dispatch ran; every file it reached recorded its own terminal state.
    Dispatched,
    /// The whole job was refused before or during dispatch.
    Refused(JobRefusal),
}

/// Host-owned orchestration seam for queued job execution policy.
///
/// This keeps the shared runner from knowing how a specific server backend
/// persists re-queues, computes backoff, or wakes its scheduler when a job
/// cannot currently run.
#[async_trait]
pub trait QueuedJobOrchestrator: Send + Sync {
    /// Handle a host-memory rejection for one queued job. The runner logs
    /// what was requested and why it was refused; the orchestrator decides
    /// only what happens to the job.
    async fn handle_memory_gate_rejection(
        &self,
        sink: &Arc<dyn RunnerEventSink>,
    ) -> Result<MemoryGateRejectionDisposition, crate::error::ServerError>;
}
