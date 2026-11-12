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
///
/// It carries the host its pool was prepared on, and an app or direct host
/// built over it takes its host from here, never from a second argument: an
/// app judging memory against one machine while its workers admit model
/// memory against another is unrepresentable.
#[derive(Clone)]
pub struct PreparedWorkers {
    pool: Arc<WorkerPool>,
    test_echo_mode: bool,
    host: Arc<crate::host_facts::HostSnapshot>,
}

/// One host-neutral execution runtime resolved from prepared workers.
pub(crate) struct ResolvedExecutionRuntime {
    pub capability_snapshot: WorkerCapabilitySnapshot,
    pub engine: ExecutionEngine,
}

impl PreparedWorkers {
    /// The host this pool was prepared on. Production reads it only
    /// through [`Self::host_for`], which resolves it with an app's
    /// configuration.
    #[cfg(test)]
    pub(crate) fn host(&self) -> &Arc<crate::host_facts::HostSnapshot> {
        &self.host
    }

    /// The host an app or direct host configured by `config` runs on, over
    /// these workers: this pool's machine, resolved with that configuration.
    /// Refused when the configuration would have the app keep a different
    /// reserve, startup limit or tier than the workers it dispatches to
    /// (one machine judged two ways), since the pool cannot follow it.
    pub fn host_for(
        &self,
        config: &crate::config::ServerConfig,
    ) -> Result<Arc<crate::host_facts::HostSnapshot>, crate::error::ServerError> {
        let host = self.host.reconfigured(config);
        if host.runtime_host() != self.host.runtime_host() {
            return Err(crate::error::ServerError::Validation(
                "this configuration resolves a different host than its workers were prepared on \
                 (memory reserve, startup limit or tier); prepare workers on this configuration"
                    .to_owned(),
            ));
        }
        Ok(Arc::new(host))
    }

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
/// or from the job runner's per-job `pre_scale_for_command`.
///
/// Capabilities are detected lazily on the first real worker spawn rather than
/// at startup, which avoids a 10-30 second delay and a 2-3 GB peak from a probe
/// worker on small machines.
///
/// `host` is the one reading of the machine this pool and everything built
/// over it decide by. A pool configuration whose runtime reads another host
/// is refused here, at composition, rather than left to admit memory the app
/// never offered.
pub async fn prepare_workers(
    host: Arc<crate::host_facts::HostSnapshot>,
    pool_config: PoolConfig,
    discovery: RegistryDiscovery,
) -> Result<PreparedWorkers, crate::error::ServerError> {
    if pool_config.runtime.host != host.runtime_host() {
        return Err(crate::error::ServerError::Validation(
            "the worker pool's runtime reads a different host than the one it is prepared on"
                .to_owned(),
        ));
    }
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
        host,
    };
    info!(
        capabilities = ?prepared.capability_snapshot().commands(),
        detected = prepared.pool.detected_capabilities().is_some(),
        "Prepared worker capability view"
    );
    Ok(prepared)
}

/// Echo workers behind the REAL job runner, for tests of runner behaviour
/// (cancellation, file-task supervision) at the daemon HTTP boundary.
///
/// [`prepare_workers`] couples two things for a test-echo pool: the workers
/// answer with placeholders, AND the runner short-circuits every job before
/// dispatch, because a placeholder must never be read as a result. A test of
/// what the runner does WHILE a dispatch is in flight needs the first without
/// the second; it must never read a result, and it is refused a pool whose
/// workers are real.
#[doc(hidden)]
pub async fn prepare_echo_workers_behind_live_runner(
    host: Arc<crate::host_facts::HostSnapshot>,
    pool_config: PoolConfig,
) -> Result<PreparedWorkers, crate::error::ServerError> {
    if !pool_config.test_echo {
        return Err(crate::error::ServerError::Validation(
            "a live runner over echo workers needs a test-echo pool".to_owned(),
        ));
    }
    let prepared = prepare_workers(host, pool_config, RegistryDiscovery::Ignore).await?;
    Ok(PreparedWorkers {
        test_echo_mode: false,
        ..prepared
    })
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::api::MemoryMb;
    use crate::config::ServerConfig;
    use crate::host_facts::HostSnapshot;

    /// A test-echo pool configured to run on `host`.
    fn echo_pool_on(host: &HostSnapshot) -> PoolConfig {
        PoolConfig {
            test_echo: true,
            runtime: crate::worker::handle::WorkerRuntimeConfig::for_tests_on(host),
            ..PoolConfig::for_tests()
        }
    }

    /// A configuration keeping no memory headroom free, where the default
    /// keeps some: one knob that changes the host's runtime half.
    fn no_headroom() -> ServerConfig {
        ServerConfig {
            memory_gate_mb: Some(MemoryMb(0)),
            ..ServerConfig::default()
        }
    }

    /// A pool whose runtime reads another host than the one it is prepared
    /// on is refused at composition, so the two cannot disagree later.
    #[tokio::test]
    async fn a_pool_on_another_host_is_refused_at_composition() {
        let prepared_on = Arc::new(HostSnapshot::for_tests(&no_headroom()));
        let elsewhere = HostSnapshot::for_tests(&ServerConfig::default());
        let refused = prepare_workers(
            prepared_on.clone(),
            echo_pool_on(&elsewhere),
            RegistryDiscovery::Ignore,
        )
        .await;
        assert!(matches!(
            refused,
            Err(crate::error::ServerError::Validation(_))
        ));

        let prepared = prepare_workers(
            prepared_on.clone(),
            echo_pool_on(&prepared_on),
            RegistryDiscovery::Ignore,
        )
        .await
        .expect("a pool on its own host");
        assert!(Arc::ptr_eq(prepared.host(), &prepared_on));
    }

    /// An app over prepared workers takes their machine, resolved with its
    /// own configuration, and is refused a configuration that would keep a
    /// different reserve than the workers it dispatches to.
    #[tokio::test]
    async fn an_app_configuration_must_resolve_the_workers_host() {
        let host = Arc::new(HostSnapshot::for_tests(&no_headroom()));
        let workers = prepare_workers(host.clone(), echo_pool_on(&host), RegistryDiscovery::Ignore)
            .await
            .expect("prepared");
        let same = workers.host_for(&no_headroom()).expect("same reserve");
        assert_eq!(same.runtime_host(), host.runtime_host());
        assert!(matches!(
            workers.host_for(&ServerConfig::default()),
            Err(crate::error::ServerError::Validation(_))
        ));
    }
}
