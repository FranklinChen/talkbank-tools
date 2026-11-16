//! PID file reaper: garbage-collects orphaned Python worker processes.
//!
//! # Problem
//!
//! Worker processes are spawned with `setpgid(0, 0)` (their own process group)
//! so they survive the death of the parent Rust server. If the server is killed
//! (SIGKILL, OOM, test runner exit), `Drop` impls never fire and workers
//! accumulate indefinitely, consuming RAM.
//!
//! # Design
//!
//! Three atomic operations, no locking:
//!
//! 1. **On spawn:** Write `<state-dir>/worker-pids/{worker_pid}` containing
//!    the server PID and a timestamp.
//! 2. **On worker shutdown/drop:** Remove the PID file.
//! 3. **On pool startup:** Scan the directory. For each file where the worker
//!    is alive but the recorded server is dead → orphan → SIGTERM + SIGKILL.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;

use tracing::{debug, info, warn};

/// Directory name under the configured batchalign3 state directory.
use crate::worker::process_id::NativePid;

const PID_DIR_NAME: &str = "worker-pids";

fn pid_dir_from_layout(layout: &crate::config::RuntimeLayout) -> PathBuf {
    layout.state_dir().join(PID_DIR_NAME)
}

/// Return the PID file directory under the resolved runtime state directory.
fn pid_dir() -> PathBuf {
    pid_dir_from_layout(&crate::config::RuntimeLayout::from_env())
}

/// Ensure the PID file directory exists.
fn ensure_pid_dir() -> Option<PathBuf> {
    let dir = pid_dir();
    if !dir.exists()
        && let Err(e) = fs::create_dir_all(&dir)
    {
        warn!(path = %dir.display(), error = %e, "Failed to create worker PID directory");
        return None;
    }
    Some(dir)
}

/// Record a spawned worker by writing its PID file.
///
/// File format: `server_pid={server_pid}\n`, one line, easy to parse.
pub(crate) fn record_worker_pid(worker_pid: u32) {
    let Some(dir) = ensure_pid_dir() else {
        return;
    };
    let path = dir.join(worker_pid.to_string());
    let server_pid = std::process::id();
    let content = format!("server_pid={server_pid}\n");
    match fs::File::create(&path).and_then(|mut f| f.write_all(content.as_bytes())) {
        Ok(()) => {
            debug!(worker_pid, server_pid, path = %path.display(), "Recorded worker PID file");
        }
        Err(e) => {
            warn!(worker_pid, path = %path.display(), error = %e, "Failed to write worker PID file");
        }
    }
}

/// Remove a worker's PID file (called on clean shutdown / Drop).
pub(crate) fn remove_worker_pid(worker_pid: u32) {
    let dir = pid_dir();
    let path = dir.join(worker_pid.to_string());
    match fs::remove_file(&path) {
        Ok(()) => {
            debug!(worker_pid, path = %path.display(), "Removed worker PID file");
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            // Already cleaned up: fine.
        }
        Err(e) => {
            warn!(worker_pid, path = %path.display(), error = %e, "Failed to remove worker PID file");
        }
    }
}

/// Scan the PID directory and kill orphaned workers.
///
/// An orphan is a worker whose PID file exists, whose process is still alive,
/// but whose recorded server PID is dead (server crashed or exited).
///
/// Returns the number of orphans reaped.
pub(crate) fn reap_orphaned_workers() -> usize {
    let dir = pid_dir();
    if !dir.exists() {
        return 0;
    }

    let entries = match fs::read_dir(&dir) {
        Ok(entries) => entries,
        Err(e) => {
            warn!(path = %dir.display(), error = %e, "Failed to read worker PID directory");
            return 0;
        }
    };

    let mut reaped = 0;
    let current_server_pid = std::process::id();

    for entry in entries.flatten() {
        let path = entry.path();
        let Some(filename) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        let Ok(worker_pid) = filename.parse::<u32>() else {
            // Not a PID file, skip.
            continue;
        };

        match classify_pid_file(&path, worker_pid, current_server_pid) {
            PidFileStatus::Stale => {
                // Worker is dead, just clean up the file.
                debug!(worker_pid, "Removing stale PID file (worker already dead)");
                let _ = fs::remove_file(&path);
            }
            PidFileStatus::Orphan { server_pid, worker } => {
                info!(
                    worker_pid,
                    dead_server_pid = server_pid.get(),
                    "Reaping orphaned worker (parent server is dead)"
                );
                kill_orphan(worker);
                let _ = fs::remove_file(&path);
                reaped += 1;
            }
            PidFileStatus::OwnedByUs => {
                // This worker belongs to our server, leave it alone.
                debug!(worker_pid, "PID file belongs to current server, skipping");
            }
            PidFileStatus::OwnedByOtherLiveServer { server_pid } => {
                // Another live server owns this worker, leave it alone.
                debug!(
                    worker_pid,
                    other_server_pid = server_pid.get(),
                    "PID file belongs to another live server, skipping"
                );
            }
            PidFileStatus::Unreadable => {
                // Can't parse the file, remove it to avoid accumulation.
                warn!(worker_pid, "Removing unreadable PID file");
                let _ = fs::remove_file(&path);
            }
        }
    }

    if reaped > 0 {
        info!(reaped, "Reaped orphaned worker(s)");
    }
    reaped
}

/// Classification of a PID file.
enum PidFileStatus {
    /// Worker process is dead. File is stale.
    Stale,
    /// Worker is alive but its server is dead, orphan.
    Orphan {
        server_pid: NativePid,
        worker: NativePid,
    },
    /// Worker belongs to the current server process.
    OwnedByUs,
    /// Worker belongs to another server that is still alive.
    OwnedByOtherLiveServer { server_pid: NativePid },
    /// PID file could not be parsed.
    Unreadable,
}

fn classify_pid_file(path: &Path, worker_pid: u32, current_server_pid: u32) -> PidFileStatus {
    // Read and parse the PID file.
    let content = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return PidFileStatus::Unreadable,
    };

    let server_pid = match parse_server_pid(&content) {
        Some(pid) => pid,
        None => return PidFileStatus::Unreadable,
    };

    // Is the worker still alive?
    let Some(worker) = NativePid::admit(worker_pid) else {
        return PidFileStatus::Unreadable;
    };
    if !worker.is_alive() {
        return PidFileStatus::Stale;
    }

    // Worker is alive. Is its server still alive?
    if server_pid.get() == current_server_pid {
        return PidFileStatus::OwnedByUs;
    }

    if server_pid.is_alive() {
        return PidFileStatus::OwnedByOtherLiveServer { server_pid };
    }

    PidFileStatus::Orphan { server_pid, worker }
}

/// Parse `server_pid=12345` from the PID file content.
fn parse_server_pid(content: &str) -> Option<NativePid> {
    for line in content.lines() {
        let trimmed = line.trim();
        if let Some(value) = trimmed.strip_prefix("server_pid=") {
            return value.trim().parse::<u32>().ok().and_then(NativePid::admit);
        }
    }
    None
}

/// Legacy raw-ID boundary. A non-process selector cannot acquire native authority.
pub(crate) fn process_alive(pid: u32) -> bool {
    NativePid::admit(pid).is_some_and(NativePid::is_alive)
}

fn signal_target(pid: u32) -> Option<NativePid> {
    let target = NativePid::admit(pid);
    if target.is_none() {
        tracing::warn!(pid, "Refusing invalid native worker process ID");
    }
    target
}
pub(crate) fn terminate_pgid(pid: u32) {
    if let Some(target) = signal_target(pid) {
        target.terminate();
    }
}
pub(crate) fn kill_pgid(pid: u32) {
    if let Some(target) = signal_target(pid) {
        target.kill();
    }
}

/// Kill an orphaned worker: SIGTERM, wait 2s, SIGKILL.
#[cfg(unix)]
fn kill_orphan(worker: NativePid) {
    worker.terminate();
    std::thread::sleep(Duration::from_secs(2));
    if worker.is_alive() {
        info!(
            worker_pid = worker.get(),
            "Orphan didn't exit after SIGTERM, sending SIGKILL"
        );
        worker.kill();
    }
}

#[cfg(not(unix))]
fn kill_orphan(_worker_pid: NativePid) {
    // No-op on non-Unix platforms. Workers on Windows don't use setpgid
    // and are tied to the parent process.
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Guards: PID marker bytes retain their actual positive owner ID and
    /// refuse malformed, zero and out-of-range native process selectors.
    /// Kind: wire
    /// Not a type: external marker text and numeric range must earn NativePid
    /// admission before any process/group syscall can consume the result.
    /// State: immutable strings exercise the real marker parser and target admission.
    /// Concurrency: none
    #[test]
    fn marker_bytes_admit_only_native_process_ids() {
        for (bytes, expected) in [
            ("server_pid=12345\n", Some(12345)),
            ("server_pid=1\n", Some(1)),
            ("  server_pid=99999  \n", Some(99999)),
            ("", None),
            ("garbage", None),
            ("server_pid=notanumber\n", None),
            ("server_pid=0", None),
        ] {
            assert_eq!(parse_server_pid(bytes).map(NativePid::get), expected);
        }
        #[cfg(unix)]
        assert!(NativePid::admit(u32::MAX).is_none());
    }

    /// Guards: PID records use the selected state root's worker-pids child.
    /// Kind: policy
    /// Not a type: the externally recognized directory name is a layout choice
    /// shared with cleanup, after RuntimeLayout owns the admitted source root.
    /// State: local path values require no directory or environment mutation.
    /// Concurrency: none
    #[test]
    fn pid_dir_is_under_configured_state_root() {
        let layout = crate::config::RuntimeLayout::from_state_dir(PathBuf::from("/tmp/ba3-state"));
        assert_eq!(
            pid_dir_from_layout(&layout),
            PathBuf::from("/tmp/ba3-state/worker-pids")
        );
    }

    /// Guards: native liveness observes the owned current process and rejects
    /// a raw zero selector instead of interpreting it as the current group.
    /// Kind: boundary
    /// Not a type: syscall existence results are kernel observations after native-ID
    /// admission. The legacy wrapper must actually use that admission edge.
    /// State: one signal-zero query observes this test process without signalling it.
    /// Concurrency: the current process owns its PID and no environment or process group changes.
    #[cfg(unix)]
    #[test]
    fn native_liveness_observes_the_admitted_process() {
        assert!(process_alive(std::process::id()));
        assert!(!process_alive(0));
    }

    /// Guards: PID zero is refused before signalling, so an invalid worker ID
    /// cannot become a signal to the caller's entire process group.
    /// Kind: boundary
    /// Not a type: the legacy raw-PID entry must actually use native-ID admission
    /// before its kernel effect. The child process observes that external boundary.
    /// State: a child running this exact test has its own process group, isolating
    /// the original zero-target failure from the controller and other processes.
    /// Concurrency: only the child receives the probe environment. Its isolated
    /// group contains no other task, and the parent waits for its actual exit.
    #[cfg(unix)]
    #[test]
    fn invalid_signal_ids_never_reach_the_kernel() {
        use std::os::unix::process::CommandExt;
        const PROBE: &str = "BA3_REAPER_PID_PROBE";
        if std::env::var_os(PROBE).as_deref() == Some(std::ffi::OsStr::new("isolated")) {
            terminate_pgid(0);
            kill_pgid(0);
            return;
        }
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "worker::pool::reaper::tests::invalid_signal_ids_never_reach_the_kernel",
                "--nocapture",
            ])
            .env(PROBE, "isolated")
            .process_group(0)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "isolated PID-zero probe exited {:?}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
