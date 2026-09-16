//! Worker pool preparation for both direct and server execution hosts.
//!
//! This module owns pool construction and the capability view shared between
//! direct-mode CLI execution and the HTTP server. It does NOT depend on axum,
//! sqlx, or any server-specific crate.

use std::sync::Arc;

use tracing::info;

use crate::cache::UtteranceCache;
use crate::capability::WorkerCapabilitySnapshot;
use crate::runner::{ExecutionEngine, RunnerExecutionContext};
use crate::worker::InferTask;
use crate::worker::pool::{PoolConfig, WorkerPool};

// ---------------------------------------------------------------------------
// Prepared worker subsystem
// ---------------------------------------------------------------------------

/// Prepared worker subsystem that can be reused across multiple app instances.
///
/// Tests use this seam to amortize capability probing while still creating a
/// fresh control plane and runtime-owned filesystem layout for each isolated
/// session.
///
/// It holds no capability copy of its own: the view is resolved from the
/// pool's admitted report whenever it is asked for, so it cannot go stale
/// against a report that arrived after preparation.
#[derive(Clone)]
pub struct PreparedWorkers {
    pool: Arc<WorkerPool>,
    test_echo_mode: bool,
}

/// One host-neutral execution runtime resolved from prepared workers.
pub(crate) struct ResolvedExecutionRuntime {
    pub capability_snapshot: WorkerCapabilitySnapshot,
    pub engine: ExecutionEngine,
}

impl PreparedWorkers {
    /// The current capability view: every released command until a worker
    /// has answered (and always for test-echo workers), then the first
    /// admitted report's commands.
    pub(crate) fn capability_snapshot(&self) -> WorkerCapabilitySnapshot {
        WorkerCapabilitySnapshot::resolve(
            self.test_echo_mode,
            self.pool.detected_capabilities().map(Arc::as_ref),
        )
    }

    /// The infer tasks behind the current capability view.
    pub fn infer_tasks(&self) -> Vec<InferTask> {
        self.capability_snapshot().infer_tasks().to_vec()
    }

    /// Build one host-neutral execution runtime over this prepared worker set.
    pub(crate) fn resolve_execution_runtime(
        &self,
        cache: Arc<UtteranceCache>,
    ) -> ResolvedExecutionRuntime {
        ResolvedExecutionRuntime {
            capability_snapshot: self.capability_snapshot(),
            engine: ExecutionEngine::new(RunnerExecutionContext::new(
                self.pool.clone(),
                cache,
                self.test_echo_mode,
            )),
        }
    }

    /// Return a reference to the underlying worker pool.
    ///
    /// Exposed for server-only code that needs direct pool access (e.g.,
    /// building [`AppState`](crate::state::AppState)).
    pub fn pool(&self) -> &Arc<WorkerPool> {
        &self.pool
    }
}

// ---------------------------------------------------------------------------
// Worker probing and preparation
// ---------------------------------------------------------------------------

/// Whether pool preparation adopts TCP workers already listed in the registry.
///
/// A named choice rather than a `bool` parameter: the two call sites differ on
/// a real policy question, not on a flag. Server preparation adopts
/// pre-started daemons; direct inline execution deliberately does not, so a
/// one-shot CLI run never inherits a detached daemon it did not start and will
/// not retire.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegistryDiscovery {
    /// Adopt TCP workers found in the registry file.
    Adopt,
    /// Ignore the registry; use only workers this process creates.
    Ignore,
}

/// Build the worker pool.
///
/// The returned [`PreparedWorkers`] value owns a live [`WorkerPool`]. Callers
/// can share that value across multiple app instances to keep expensive model
/// loads hot while still rebuilding the server control plane and
/// runtime-owned temp directories.
///
/// Returns without spawning a worker. Workers arrive either from the registry
/// (a pre-started TCP daemon, adopted when `discovery` is
/// [`RegistryDiscovery::Adopt`], whose report the pool admits as it probes it)
/// or from the job runner's per-job `pre_scale_for_command_options`.
///
/// Capabilities are detected lazily on the first real worker spawn rather than
/// at startup, which avoids a 10-30 second delay and a 2-3 GB peak from a probe
/// worker on small machines.
pub async fn prepare_workers(
    pool_config: PoolConfig,
    discovery: RegistryDiscovery,
) -> Result<PreparedWorkers, crate::error::ServerError> {
    let test_echo_mode = pool_config.test_echo;
    let pool = Arc::new(WorkerPool::new(pool_config));
    // `None` only for a pool with no runtime, which `prepare_workers` (an
    // `async fn`) can never produce.
    let _ = pool.start_background_tasks();

    if discovery == RegistryDiscovery::Adopt {
        let discovered = pool.discover_from_registry().await;
        if discovered > 0 {
            info!(discovered, "Pre-started TCP workers integrated into pool");
        }
    }

    let prepared = PreparedWorkers {
        pool,
        test_echo_mode,
    };
    info!(
        capabilities = ?prepared.capability_snapshot().commands(),
        detected = prepared.pool.detected_capabilities().is_some(),
        "Prepared worker capability view"
    );
    Ok(prepared)
}
