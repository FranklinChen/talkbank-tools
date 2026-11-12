//! Process-global memory guard for Python worker spawning, and the typed
//! refusals of every host-memory wait a worker makes.
//!
//! # Problem
//!
//! Python ML workers load gigabytes of models (Whisper, Stanza, etc.). When
//! several start or load at once (parallel tests, pre-scaling, job dispatch)
//! each sees "enough" memory before any of them has taken its share, and by
//! the time all are resident they exceed physical RAM, triggering a
//! **kernel-level OOM panic** that crashes the entire machine (not just the
//! process).
//!
//! This has caused multiple catastrophic crashes on a 64 GB developer machine
//! (see `docs/postmortems/`). The Jetsam report from 2026-03-21 showed 5
//! python3.12 workers at 13-15 GB each = 71 GB on a 64 GB machine.
//!
//! # Solution
//!
//! Every byte a worker holds is reserved with the host-wide ledger
//! (`crate::host_memory`) before it is taken: the process baseline by the
//! spawn's startup lease, here, and each model by its load's own lease
//! (`worker::pool::task_load`). A process-global semaphore also serializes
//! spawns within one process, and the ledger's startup slots across the
//! host's batchalign3 processes.
//!
//! A reservation the host cannot fit now is queued, not refused: the spawn
//! waits, reported, until the host can fit it, and a cancelled job stops the
//! wait by dropping it. Only a reservation that can never fit, or a host or
//! ledger that cannot be read, is refused, with its own typed error.

use std::sync::LazyLock;

use tokio::sync::Semaphore;
use tracing::warn;

use crate::host_memory::{HostMemoryLease, QueuedLeaseFailure, WorkerLeaseRequest};

use super::handle::WorkerConfig;

/// Maximum concurrent worker spawns. Serializing spawns prevents the TOCTOU
/// race where N workers all check memory before any of them has started.
///
/// Default: 1 (fully serialized). Override via `BATCHALIGN_MAX_CONCURRENT_SPAWNS`.
fn max_concurrent_spawns() -> usize {
    std::env::var("BATCHALIGN_MAX_CONCURRENT_SPAWNS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1)
}

/// Process-global spawn semaphore.
static SPAWN_SEMAPHORE: LazyLock<Semaphore> =
    LazyLock::new(|| Semaphore::new(max_concurrent_spawns()));

/// Why a worker's host-memory reservation (a spawn's baseline, a model's
/// load, a per-batch build) was refused.
///
/// Each cause keeps its own type so the job-failure classifier
/// (`runner::util::error_classification`) can tell a reservation no host
/// state will ever fit from a host or ledger the coordinator cannot use.
/// There is no "not enough memory now": that wait queues.
#[derive(Debug, thiserror::Error)]
pub enum MemoryGuardError {
    /// Spawn semaphore was closed (server shutting down).
    #[error("worker spawn semaphore closed (server shutting down)")]
    SemaphoreClosed,

    /// The reservation is larger than this host can ever grant with its
    /// reserve kept free: no amount of waiting fits it.
    #[error(transparent)]
    ExceedsHost(crate::host_memory::NeverFits),

    /// The coordinator could not read the host or use its ledger.
    #[error("host memory coordinator could not decide a reservation: {0}")]
    LedgerUnusable(#[source] crate::host_memory::UnusableHost),

    /// The blocking task taking the reservation panicked or was cancelled.
    #[error("host memory reservation task did not finish: {0}")]
    ReservationTaskFailed(#[source] tokio::task::JoinError),
}

impl From<QueuedLeaseFailure> for MemoryGuardError {
    /// A queued wait stops only on a final refusal, its own type: neither a
    /// congestion nor a timeout can arrive here.
    fn from(failure: QueuedLeaseFailure) -> Self {
        use crate::host_memory::FinalRefusal;
        match failure {
            QueuedLeaseFailure::Refused(FinalRefusal::NeverFits(never)) => Self::ExceedsHost(never),
            QueuedLeaseFailure::Refused(FinalRefusal::Unusable(unusable)) => {
                Self::LedgerUnusable(unusable)
            }
            QueuedLeaseFailure::AttemptDidNotFinish(join) => Self::ReservationTaskFailed(join),
        }
    }
}

/// Combined local+host guard that protects one worker startup window.
pub struct SpawnPermit {
    _local_permit: tokio::sync::SemaphorePermit<'static>,
    host_lease: HostMemoryLease,
}

impl SpawnPermit {
    /// Tag the underlying host-memory lease with the freshly-spawned
    /// worker's PID. The caller must invoke this immediately after
    /// `cmd.spawn()` succeeds: prior to the call the lease is owned
    /// (in liveness terms) by the daemon, and a daemon-PID-only lease
    /// survives worker death (Bug 2 / ghost slots, 2026-05-01).
    ///
    /// Failures to update the ledger are warned and dropped: this is a
    /// best-effort accounting hint, not a correctness gate. A worker
    /// that never gets its PID written into the lease falls back to
    /// the older "owner_pid is daemon" behaviour, which is the
    /// pre-fix status quo: strictly no worse than today.
    pub fn set_worker_pid(&self, worker_pid: crate::worker::WorkerPid) {
        if let Err(error) = self.host_lease.set_worker_pid(worker_pid) {
            warn!(
                worker_pid = worker_pid.0,
                error = %error,
                "Failed to tag host-memory lease with worker PID; ghost-slot reaping will fall back to owner_pid",
            );
        }
    }
}

/// Free permits on the process-global spawn semaphore, i.e. how many spawns
/// could start right now.
///
/// Zero means some spawn already holds the only permit and every other spawn in
/// this process is queued behind it until that worker reports ready. Exposed so
/// a stalled test can say which of "waiting for the machine" and "waiting for a
/// worker that will never register" it is looking at: a spawn that has not yet
/// acquired this permit has created NO process, so the absence of a new pid is
/// evidence of queuing, not of broken registration wiring. Diagnostic only;
/// nothing schedules off it.
pub fn available_spawn_permits() -> usize {
    SPAWN_SEMAPHORE.available_permits()
}

/// Acquire a spawn permit: the process's spawn semaphore, then the host's
/// startup lease for the worker's baseline.
///
/// This is the **only** path through which workers should be spawned. The
/// lease reserves [`crate::worker::residency::WORKER_BASELINE`], what a
/// worker holds before any load, since no model loads at startup: each model
/// takes its own lease when it loads. The lease is queued, never refused for
/// a host that is merely full (see the module docs); a cancelled job stops
/// the wait by dropping this future.
///
/// While queued it retires idle workers through `reclaim` (the pool's, or
/// [`crate::host_memory::NoIdleWorkers`] for a process with no pool).
///
/// Returns a permit guard. The caller holds the guard through the spawn's
/// ready handshake, so the next spawn's admission sees this one's baseline
/// either reserved or already taken.
pub async fn acquire_spawn_permit(
    config: &WorkerConfig,
    reclaim: &dyn crate::host_memory::MemoryReclaim,
) -> Result<SpawnPermit, MemoryGuardError> {
    let permit = SPAWN_SEMAPHORE
        .acquire()
        .await
        .map_err(|_| MemoryGuardError::SemaphoreClosed)?;

    let request =
        WorkerLeaseRequest::worker_startup(config.profile, &config.lang, &config.engine_overrides);
    let observer = super::pool::checkout_wait::MemoryWaitObserver::current();
    // Drops the semaphore permit on refusal (`?` returns before it is held),
    // so other spawns can proceed.
    let host_lease = config
        .runtime
        .host
        .memory_coordinator()
        .queue_for_lease(request, &observer, reclaim)
        .await?;

    Ok(SpawnPermit {
        _local_permit: permit,
        host_lease,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{LanguageCode3, MemoryMb, WorkerLanguage};
    use crate::host_memory::HostMemoryRuntimeConfig;
    use crate::worker::WorkerProfile;
    use tempfile::tempdir;

    /// A reserve no host can keep free refuses the spawn at once, naming the
    /// sizes, rather than queueing it for ever.
    #[tokio::test]
    async fn a_spawn_no_host_can_fit_is_refused_at_once() {
        let temp = tempdir().expect("tempdir");
        let config = WorkerConfig {
            profile: WorkerProfile::Stanza,
            lang: WorkerLanguage::from(LanguageCode3::eng()),
            runtime: crate::worker::handle::WorkerRuntimeConfig {
                host: crate::worker::handle::WorkerRuntimeConfig::for_tests()
                    .host
                    .with_ledger(HostMemoryRuntimeConfig::from_sources(
                        temp.path().join("host-memory.json"),
                        MemoryMb(1_000_000_000),
                        crate::config::WorkerStartupLimit::literal::<1>(),
                    )),
                ..crate::worker::handle::WorkerRuntimeConfig::for_tests()
            },
            ..WorkerConfig::for_tests()
        };
        let result = acquire_spawn_permit(&config, &crate::host_memory::NoIdleWorkers).await;
        assert!(
            matches!(
                result,
                Err(MemoryGuardError::ExceedsHost(
                    crate::host_memory::NeverFits {
                        reserve_mb: 1_000_000_000,
                        ..
                    }
                ))
            ),
            "expected a refusal no waiting clears"
        );
    }
}
