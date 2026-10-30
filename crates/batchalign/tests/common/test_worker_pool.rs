//! Per-test-binary, ConfigKey-keyed Python-worker fixture.
//!
//! Each `cargo test` run of the worker-spawning suites fork/exec's
//! ~50 `uv run python -m batchalign.worker` processes. At default parallelism
//! the spawn tail under contention exceeded `ready_timeout_s`, producing
//! flaky failures. This fixture is the principled fix: tests within a binary
//! share workers keyed by the subset of [`WorkerConfig`] the Python child
//! observes at startup.
//!
//! A runner config used to cap that parallelism too. That runner is banned
//! and uninstalled here, so nothing caps it now, which is why the suites
//! rely on this fixture rather than on a cap.
//!
//! Fields the Python child does NOT observe, `ready_timeout_s`,
//! `audio_task_timeout_s`, `analysis_task_timeout_s`, `runtime.host`,
//! `runtime.server_instance_id`,
//! `runtime.server_process_id`: are deliberately excluded from
//! [`ConfigKey`], so tests that vary only those fields still share a worker.
//! Source of truth for the observed/excluded split is
//! `crate::worker::handle::spawn::build_worker_command`; if a flag or env
//! var is added there, mirror it in [`ConfigKey`].
//!
//! Workers live for the test-binary process lifetime; the `OnceLock` is
//! never dropped, so on process exit the workers become orphans and are
//! reaped on the next batchalign run via the worker registry's PID-file
//! mechanism. Acceptable for `--test-echo` workers (no model state).
//!
//! # Why the workers get their own runtime
//!
//! A pooled worker outlives the test that happened to spawn it, but
//! `#[tokio::test]` gives every test its OWN runtime and drops it at the end
//! of that test. A `WorkerHandle` owns a `tokio::process::Child` whose I/O is
//! registered with the reactor of the runtime that created it, so spawning
//! under a per-test runtime left every later test holding a handle whose
//! reactor was gone:
//!
//! ```text
//! A Tokio 1.x context was found, but it is being shutdown.
//! ```
//!
//! That failed 5 to 6 of `worker_integration`'s tests depending on which test
//! initialised the pool first, so it looked order-dependent and intermittent
//! rather than structural (found 2026-07-29, after two unrelated failures
//! stopped masking this binary).
//!
//! Workers are therefore spawned ON [`worker_runtime()`], a static
//! multi-thread runtime that is never dropped. Its reactor outlives every
//! test, so a handle stays usable no matter which test's runtime later awaits
//! it.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use tokio::sync::{Mutex, OnceCell, OwnedMutexGuard};

use batchalign::api::{NumSpeakers, WorkerLanguage};
use batchalign::worker::error::WorkerError;
use batchalign::worker::handle::{WorkerConfig, WorkerHandle};
use batchalign::worker::{InferTask, WorkerBootstrapMode, WorkerProfile};

/// The subset of [`WorkerConfig`] that determines Python-side worker
/// behavior. Two configs with the same `ConfigKey` produce equivalent
/// child processes and may safely share a single worker.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct ConfigKey {
    python_path: String,
    profile: WorkerProfile,
    task: Option<InferTask>,
    lang: WorkerLanguage,
    num_speakers: NumSpeakers,
    engine_overrides: String,
    test_echo: bool,
    verbose: u8,
    force_cpu: bool,
    gpu_thread_pool_size: u32,
    bootstrap_mode: WorkerBootstrapMode,
    revai_api_key: Option<String>,
    test_delay_ms: u64,
    /// Two configs with different state dirs are not interchangeable; see
    /// `WorkerRuntimeConfig::state_dir`.
    state_dir: Option<std::path::PathBuf>,
}

impl ConfigKey {
    fn from_config(config: &WorkerConfig) -> Self {
        Self {
            python_path: config.python_path.clone(),
            profile: config.profile,
            task: config.task,
            lang: config.lang.clone(),
            num_speakers: config.num_speakers,
            engine_overrides: config.engine_overrides.clone(),
            test_echo: config.test_echo,
            verbose: config.verbose,
            force_cpu: config.runtime.force_cpu,
            gpu_thread_pool_size: config.runtime.gpu_thread_pool_size,
            bootstrap_mode: config.runtime.bootstrap_mode,
            revai_api_key: config
                .runtime
                .revai_api_key
                .as_ref()
                .map(|key| key.as_str().to_owned()),
            test_delay_ms: config.test_delay_ms,
            state_dir: config.runtime.state_dir.clone(),
        }
    }
}

/// One pool entry: the worker handle is initialized once on first
/// checkout for this key, and serialized for dispatch thereafter.
struct WorkerCell {
    handle: OnceCell<Arc<Mutex<WorkerHandle>>>,
}

/// Per-binary fixture.
pub struct SharedTestWorkerPool {
    cells: Mutex<HashMap<ConfigKey, Arc<WorkerCell>>>,
}

impl SharedTestWorkerPool {
    fn new() -> Self {
        Self {
            cells: Mutex::new(HashMap::new()),
        }
    }

    /// Lease a worker for the given config. If a worker exists for the
    /// matching `ConfigKey`, this awaits exclusive access to it. Otherwise
    /// a new worker is spawned, inserted into the pool, and leased.
    pub async fn checkout(&self, config: &WorkerConfig) -> Result<WorkerLease, WorkerError> {
        let key = ConfigKey::from_config(config);

        // Outer lock is held only long enough to insert an empty
        // `WorkerCell`. Spawning happens under the per-key `OnceCell`,
        // so a slow Python startup for one key never blocks unrelated
        // keys from looking up or initializing their own cells.
        let cell = {
            let mut cells = self.cells.lock().await;
            cells
                .entry(key)
                .or_insert_with(|| {
                    Arc::new(WorkerCell {
                        handle: OnceCell::new(),
                    })
                })
                .clone()
        };

        let handle_arc = cell
            .handle
            .get_or_try_init(|| async {
                // Spawn ON the shared runtime, not on the caller's per-test
                // one: the child's I/O registers with whichever reactor
                // creates it, and a per-test reactor dies at end of test
                // while the pooled handle lives on. `JoinHandle` is awaitable
                // from any runtime, so the caller still just awaits here.
                let config = config.clone();
                let handle = worker_runtime()
                    .spawn(async move { WorkerHandle::spawn(config).await })
                    .await
                    .map_err(|e| {
                        WorkerError::Io(std::io::Error::other(format!(
                            "shared worker-spawn task failed: {e}"
                        )))
                    })??;
                Ok::<_, WorkerError>(Arc::new(Mutex::new(handle)))
            })
            .await?
            .clone();

        // Per-worker mutex: serializes dispatch on the one stdin/stdout
        // pipe that talks to a single Python child.
        let guard = handle_arc.lock_owned().await;
        Ok(WorkerLease { guard })
    }
}

/// Exclusive lease on one pooled worker. Drops back into the pool on
/// scope exit. Derefs (mut) to [`WorkerHandle`] so callers use it
/// exactly like a directly-spawned handle.
pub struct WorkerLease {
    guard: OwnedMutexGuard<WorkerHandle>,
}

impl std::ops::Deref for WorkerLease {
    type Target = WorkerHandle;

    fn deref(&self) -> &WorkerHandle {
        &self.guard
    }
}

impl std::ops::DerefMut for WorkerLease {
    fn deref_mut(&mut self) -> &mut WorkerHandle {
        &mut self.guard
    }
}

/// Per-binary singleton accessor. Each integration-test target compiled
/// as its own crate gets its own static area, so this is automatically
/// scoped per binary without thread-locals or external keying.
pub fn shared_test_worker_pool() -> &'static SharedTestWorkerPool {
    static POOL: OnceLock<SharedTestWorkerPool> = OnceLock::new();
    POOL.get_or_init(SharedTestWorkerPool::new)
}

/// Process-lifetime runtime that owns every pooled worker's child process.
///
/// Held in a `static`, which Rust never drops, so the reactor outlives every
/// test that borrows a worker. See the module-level note for the failure that
/// prevents. Must be multi-thread: a current-thread runtime only drives its
/// reactor inside `block_on`, and nothing ever blocks on this one, so the
/// child pipes would never be polled.
///
/// Two worker threads, not the `available_parallelism()` default: the entire
/// job is driving a handful of pipes plus a few spawns, and four test binaries
/// use this fixture concurrently.
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
