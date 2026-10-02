//! The one cross-process exclusive file lock, for read-modify-write of state
//! files and for single-writer slots.
//!
//! An OS lock (`std::fs::File::lock`, `flock` on Unix) on a dedicated lock
//! file beside the state it guards, held while a [`HeldFileLock`] lives.
//! Dedicated, never the state file itself: the state files are replaced by
//! an atomic rename (`crate::atomic_file`), and a lock taken on the file
//! being replaced is a lock on an inode the next writer no longer opens.
//!
//! Users: the server handshake slots (`server.pid.lock`,
//! `sidecar-server.pid.lock`), the CLI's per-profile daemon start locks, the
//! worker registry (`workers.json.lock`, shared with the Python registry
//! writers, which `flock` the same file), and the media cache's per-key
//! conversion slots.
//!
//! Lock files are empty and are created with the umask's default mode: they
//! hold nothing, and every process that must take them is the same user's.

use std::path::{Path, PathBuf};

use tracing::debug;

/// An exclusive lock on one lock file, held while this value lives; dropping
/// it releases the lock.
#[derive(Debug)]
pub(crate) struct HeldFileLock {
    path: PathBuf,
    /// Holding the descriptor is holding the lock; `Drop` releases it.
    file: std::fs::File,
}

impl HeldFileLock {
    /// Take the lock at `path`, creating its directory and the lock file as
    /// needed, and waiting while another process holds it. Only "held
    /// elsewhere" means wait; any other failure is an error.
    pub(crate) fn acquire(path: impl Into<PathBuf>) -> std::io::Result<Self> {
        let path = path.into();
        let file = Self::open(&path)?;
        match file.try_lock() {
            Ok(()) => {}
            Err(std::fs::TryLockError::WouldBlock) => {
                debug!(lock = %path.display(), "Waiting for a lock another process holds");
                file.lock()?;
            }
            Err(std::fs::TryLockError::Error(error)) => return Err(error),
        }
        Ok(Self { path, file })
    }

    /// [`Self::acquire`] where the lock file's directory must already exist:
    /// `None` when it does not. For an operation that has nothing to do in a
    /// missing directory (removing a record that cannot exist), which must
    /// not create the directory and a lock file by taking the lock.
    pub(crate) fn acquire_in_existing_directory(
        path: impl Into<PathBuf>,
    ) -> std::io::Result<Option<Self>> {
        let path = path.into();
        let file = match std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(&path)
        {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error),
        };
        match file.try_lock() {
            Ok(()) => {}
            Err(std::fs::TryLockError::WouldBlock) => {
                debug!(lock = %path.display(), "Waiting for a lock another process holds");
                file.lock()?;
            }
            Err(std::fs::TryLockError::Error(error)) => return Err(error),
        }
        Ok(Some(Self { path, file }))
    }

    /// [`Self::acquire`] off the async runtime: waiting for another
    /// process's lock must not hold a runtime worker thread.
    pub(crate) async fn acquire_async(path: impl Into<PathBuf>) -> std::io::Result<Self> {
        let path = path.into();
        crate::blocking::spawn_in_span(move || Self::acquire(path))
            .await
            .map_err(std::io::Error::other)?
    }

    /// Take the lock only if no other holder has it: `None` when it is held.
    #[cfg(test)]
    pub(crate) fn try_acquire(path: impl Into<PathBuf>) -> std::io::Result<Option<Self>> {
        let path = path.into();
        let file = Self::open(&path)?;
        match file.try_lock() {
            Ok(()) => Ok(Some(Self { path, file })),
            Err(std::fs::TryLockError::WouldBlock) => Ok(None),
            Err(std::fs::TryLockError::Error(error)) => Err(error),
        }
    }

    /// The lock file this value holds.
    #[cfg(test)]
    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    /// Open (creating it and its directory) the lock file.
    fn open(path: &Path) -> std::io::Result<std::fs::File> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(path)
    }
}

impl Drop for HeldFileLock {
    /// Release the lock explicitly before the descriptor closes.
    ///
    /// Closing alone does not release an `flock` while another reference to
    /// the same open file description exists, and a child forked by any
    /// thread holds one until it execs (close-on-exec acts only at exec). So
    /// a lock released by close stays held for that window, and a waiter
    /// that checks at once sees it held. Unlocking releases it now.
    fn drop(&mut self) {
        if let Err(error) = self.file.unlock() {
            debug!(lock = %self.path.display(), %error, "Releasing a lock failed; closing releases it");
        }
    }
}

/// The lock file guarding `state_file`: `<state_file>.lock` beside it.
pub(crate) fn lock_file_for(state_file: &Path) -> PathBuf {
    let mut lock = state_file.as_os_str().to_os_string();
    lock.push(".lock");
    PathBuf::from(lock)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The lock excludes a second holder until the first drops, and creates
    /// a missing directory rather than failing.
    #[test]
    fn a_held_lock_excludes_others_and_creates_its_directory() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("not-yet").join("state.json.lock");
        let held = HeldFileLock::acquire(&path).expect("acquire in a missing directory");
        assert_eq!(held.path(), path);
        assert!(
            HeldFileLock::try_acquire(&path).expect("try").is_none(),
            "a second holder is excluded"
        );
        drop(held);
        assert!(HeldFileLock::try_acquire(&path).expect("try").is_some());
    }

    /// Taking a lock in a missing directory for an operation that needs the
    /// directory creates nothing.
    #[test]
    fn a_lock_in_a_missing_directory_is_not_created_when_it_must_exist() {
        let dir = tempfile::tempdir().expect("tempdir");
        let missing = dir.path().join("never-made");
        let path = missing.join("slot.lock");
        assert!(
            HeldFileLock::acquire_in_existing_directory(&path)
                .expect("no error")
                .is_none()
        );
        assert!(!missing.exists(), "the directory must not be created");
    }

    /// Dropping releases the lock at once even while a forked child still
    /// shares the descriptor (any thread's fork does, until the child execs).
    /// Released by closing alone, the lock stayed held for that window, which
    /// made `a_publish_cannot_land_inside_a_removal` fail intermittently
    /// whenever another test spawned a process at that moment.
    #[cfg(unix)]
    #[test]
    fn dropping_releases_the_lock_while_a_forked_child_shares_the_descriptor() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("slot.lock");
        let held = HeldFileLock::acquire(&path).expect("acquire");
        // SAFETY: the child calls only async-signal-safe functions (`sleep`,
        // `_exit`) before it is killed below.
        let child = unsafe { libc::fork() };
        if child == 0 {
            unsafe {
                libc::sleep(30);
                libc::_exit(0);
            }
        }
        assert!(child > 0, "fork failed");
        drop(held);
        let reacquired = HeldFileLock::try_acquire(&path).expect("try");
        // SAFETY: `child` is the process forked above.
        unsafe {
            libc::kill(child, libc::SIGKILL);
            libc::waitpid(child, std::ptr::null_mut(), 0);
        }
        assert!(
            reacquired.is_some(),
            "the lock must be free once its holder drops it"
        );
    }

    #[test]
    fn the_lock_file_sits_beside_its_state_file() {
        assert_eq!(
            lock_file_for(Path::new("/state/workers.json")),
            PathBuf::from("/state/workers.json.lock")
        );
    }
}
