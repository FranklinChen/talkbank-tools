//! Read-only status projections for the worker pool.
//!
//! These methods query pool state without mutating it. Sequential worker groups
//! use `lock_recovered()` (blocking `std::sync::Mutex`). GPU worker maps use
//! `tokio::sync::Mutex` and are accessed via `.lock().await`, the methods are
//! async so they always acquire the lock and never silently report stale data.
//!
//! The `std::sync::MutexGuard` from `lock_recovered()` is `!Send`, so it must
//! be scoped in a block that ends before any `.await` point. This ensures the
//! resulting future is `Send` (required by axum handlers and tokio tasks).

use std::sync::atomic::Ordering;

use crate::api::{ReleasedCommand, WorkerLanguage};
use crate::options::CommandOptions;

use super::gpu_slot::GpuSlotState;
use super::{WorkerKey, WorkerPool, lock_recovered};

impl WorkerPool {
    /// Check if an idle worker exists for the exact key derived from command
    /// options.
    ///
    /// Used by the memory gate to skip the system memory check when reusable
    /// workers already exist -- those workers are already loaded, so no new
    /// memory allocation is needed. Checks both stdio and TCP workers.
    pub async fn has_idle_workers(
        &self,
        command: ReleasedCommand,
        lang: impl Into<WorkerLanguage>,
        options: &CommandOptions,
    ) -> bool {
        let key = WorkerKey::from_command_options(
            command,
            lang.into(),
            options,
            self.config.runtime.bootstrap_mode,
        );

        // GPU profile workers are always "available" (shared, concurrent).
        if key.target.is_concurrent() {
            // Check TCP GPU workers first.
            {
                let tcp_gpu_workers = self.gpu_tcp_workers.lock().await;
                if tcp_gpu_workers.contains_key(&key) {
                    return true;
                }
            }
            // A key whose spawn is still in flight does NOT count as having a
            // worker: this answers "can a request be served right now", and it
            // answered `false` for such a key before slots existed (the entry
            // appeared only once the spawn had succeeded). Preserved
            // deliberately.
            let gpu_workers = self.gpu_workers.lock().await;
            return gpu_workers
                .get(&key)
                .is_some_and(|slot| matches!(slot.state(), GpuSlotState::Ready(_)));
        }

        // Sequential groups use std::sync::Mutex, scope the guard so it
        // does not live across any .await (MutexGuard is !Send).
        let groups = lock_recovered(&self.groups);
        groups.get(&key).is_some_and(|group| {
            !lock_recovered(&group.idle).is_empty()
                || !lock_recovered(&group.tcp_workers).is_empty()
        })
    }

    /// Number of active workers (total across all keys, including checked-out).
    ///
    /// Counts sequential group workers, shared GPU workers, and TCP workers.
    /// The sequential group count uses an atomic per-group total, so no lock
    /// is held across the subsequent GPU lock awaits.
    pub async fn worker_count(&self) -> usize {
        // Scope the std::sync::MutexGuard so it is dropped before .await.
        let groups_count: usize = {
            let groups = lock_recovered(&self.groups);
            groups
                .values()
                .map(|g| g.total.load(Ordering::Relaxed))
                .sum()
        };
        // Keys whose spawn is in flight have no worker process yet, so they are
        // not counted: this total is compared against worker CAPS, and counting
        // a worker that does not exist would let the caps admit one too few.
        let gpu_count = self
            .gpu_workers
            .lock()
            .await
            .values()
            .filter(|slot| matches!(slot.state(), GpuSlotState::Ready(_)))
            .count();
        let tcp_gpu_count = self.gpu_tcp_workers.lock().await.len();
        groups_count + gpu_count + tcp_gpu_count
    }

    /// Active worker keys: `["profile:stanza:eng (2 total, 1 idle)", ...]`.
    ///
    /// Includes both sequential group workers and shared GPU workers.
    pub async fn worker_keys(&self) -> Vec<String> {
        // Scope the std::sync::MutexGuard so it is dropped before .await.
        let mut keys: Vec<String> = {
            let groups = lock_recovered(&self.groups);
            groups
                .iter()
                .map(|(key, group)| {
                    let total = group.total.load(Ordering::Relaxed);
                    let idle = lock_recovered(&group.idle).len();
                    let suffix = if key.engine_selection.is_none() {
                        String::new()
                    } else {
                        format!(":{}", key.engine_selection)
                    };
                    format!(
                        "{}:{}{suffix} ({total} total, {idle} idle)",
                        key.target.label(),
                        key.language,
                    )
                })
                .collect()
        };

        {
            // This one REPORTS an in-flight spawn rather than hiding it. It is
            // the operator-facing key listing, and "a worker for this key is
            // coming up" is exactly what an operator watching a cold start
            // wants to see; a key that silently appears minutes later reads as
            // the server having done nothing.
            let gpu_workers = self.gpu_workers.lock().await;
            for (key, slot) in gpu_workers.iter() {
                let suffix = if key.engine_selection.is_none() {
                    String::new()
                } else {
                    format!(":{}", key.engine_selection)
                };
                let state = match slot.state() {
                    GpuSlotState::Ready(_) => "1 total, shared",
                    GpuSlotState::Spawning => "spawning",
                    GpuSlotState::Empty => "awaiting spawn",
                    GpuSlotState::Unavailable => "unavailable, awaiting replacement",
                };
                keys.push(format!(
                    "{}:{}{suffix} ({state})",
                    key.target.label(),
                    key.language,
                ));
            }
        }

        {
            let tcp_gpu_workers = self.gpu_tcp_workers.lock().await;
            for key in tcp_gpu_workers.keys() {
                let suffix = if key.engine_selection.is_none() {
                    String::new()
                } else {
                    format!(":{}", key.engine_selection)
                };
                keys.push(format!(
                    "{}:{}{suffix} (1 total, tcp-shared)",
                    key.target.label(),
                    key.language,
                ));
            }
        }

        keys.sort();
        keys
    }

    /// Backward-compatible string summary derived from
    /// [`worker_summary_entries`]. Format:
    /// `["profile:stanza:eng:pid=1234:transport=stdio", ...]`.
    ///
    /// The `loaded_pipelines` health/websocket field is wire-typed
    /// as `Vec<String>` and consumed by the frontend in that form;
    /// this method preserves that contract. Internal callers
    /// (tests, future tooling) should prefer the typed version.
    pub async fn worker_summary(&self) -> Vec<String> {
        let mut s: Vec<String> = self
            .worker_summary_entries()
            .await
            .into_iter()
            .map(|e| e.to_string())
            .collect();
        s.sort();
        s
    }

    /// Structured per-worker summary. Source of truth for
    /// [`worker_summary`]. Reports idle sequential workers + shared
    /// GPU workers (stdio + TCP) + sequential TCP workers. Checked-out
    /// sequential workers are not listed; use `worker_count()` for
    /// full totals.
    pub async fn worker_summary_entries(&self) -> Vec<WorkerSummaryEntry> {
        let mut entries: Vec<WorkerSummaryEntry> = {
            let groups = lock_recovered(&self.groups);
            let mut v = Vec::new();
            for group in groups.values() {
                let idle = lock_recovered(&group.idle);
                for worker in idle.iter() {
                    v.push(WorkerSummaryEntry {
                        profile: worker.profile_label().to_string(),
                        lang: worker.lang().to_string(),
                        pid: worker.pid(),
                        transport: WorkerTransport::from_str(worker.transport()),
                        concurrent: false,
                    });
                }
            }
            v
        };

        {
            // Every field of an entry (profile, lang, pid, transport) is read
            // off the worker itself, and a key whose spawn is in flight has no
            // worker to read them from: there is no pid yet. So this listing
            // skips it, and `worker_keys` above is where an operator sees that
            // the key is coming up.
            let gpu_workers = self.gpu_workers.lock().await;
            for slot in gpu_workers.values() {
                let GpuSlotState::Ready(worker) = slot.state() else {
                    continue;
                };
                entries.push(WorkerSummaryEntry {
                    profile: worker.profile_label(),
                    lang: worker.lang().to_string(),
                    pid: worker.pid(),
                    transport: WorkerTransport::Stdio,
                    concurrent: true,
                });
            }
        }

        {
            let tcp_gpu_workers = self.gpu_tcp_workers.lock().await;
            for (key, worker) in tcp_gpu_workers.iter() {
                entries.push(WorkerSummaryEntry {
                    profile: key.target.label().to_string(),
                    lang: key.language.to_string(),
                    pid: worker.pid(),
                    transport: WorkerTransport::Tcp,
                    concurrent: true,
                });
            }
        }

        {
            let groups = lock_recovered(&self.groups);
            for group in groups.values() {
                let tcp = lock_recovered(&group.tcp_workers);
                for worker in tcp.iter() {
                    entries.push(WorkerSummaryEntry {
                        profile: worker.profile_label().to_string(),
                        lang: worker.lang().to_string(),
                        pid: worker.pid(),
                        transport: WorkerTransport::Tcp,
                        concurrent: false,
                    });
                }
            }
        }

        entries
    }
}

/// Transport channel for one worker process.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerTransport {
    /// stdio JSON-lines IPC (the default for spawned worker children).
    Stdio,
    /// TCP socket (registry-discovered or shared-GPU concurrent transport).
    Tcp,
}

impl WorkerTransport {
    fn from_str(s: &str) -> Self {
        match s {
            "tcp" => Self::Tcp,
            _ => Self::Stdio,
        }
    }

    /// Wire-format token for the transport, matching the format
    /// the older `Vec<String>` summary embedded as `transport=...`.
    pub fn as_wire(&self) -> &'static str {
        match self {
            Self::Stdio => "stdio",
            Self::Tcp => "tcp",
        }
    }
}

/// One row in the structured worker summary. Consumed by tests and
/// in-process tooling that previously parsed `pid=N` substrings out
/// of the string form returned by [`WorkerPool::worker_summary`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerSummaryEntry {
    /// Profile label (`profile:stanza`, `profile:gpu`, `profile:io`).
    pub profile: String,
    /// ISO 639-3 language code the worker was spawned for.
    pub lang: String,
    /// Worker process ID.
    pub pid: crate::worker::WorkerPid,
    /// IPC transport.
    pub transport: WorkerTransport,
    /// True for shared concurrent GPU workers; false for sequential pool workers.
    pub concurrent: bool,
}

impl std::fmt::Display for WorkerSummaryEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}:{}:pid={}:transport={}",
            self.profile,
            self.lang,
            self.pid,
            self.transport.as_wire(),
        )?;
        if self.concurrent {
            write!(f, ":concurrent")?;
        }
        Ok(())
    }
}
