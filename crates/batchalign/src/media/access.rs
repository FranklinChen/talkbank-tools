//! Bounded, on-demand access to potentially unresponsive media filesystems.

use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::Duration;

use tokio::sync::{Notify, oneshot};

pub(crate) const MEDIA_ACCESS_TIMEOUT: Duration = Duration::from_secs(5);
static ACCESS: LazyLock<Arc<MediaAccess>> = LazyLock::new(|| Arc::new(MediaAccess::new(4)));

/// Owns admission and outstanding operations, including timed-out OS threads.
struct MediaAccess {
    limit: usize,
    active: Mutex<Vec<PathBuf>>,
    changed: Notify,
}

impl MediaAccess {
    fn new(limit: usize) -> Self {
        Self {
            limit,
            active: Mutex::new(Vec::new()),
            changed: Notify::new(),
        }
    }

    async fn admit(
        self: &Arc<Self>,
        root: &Path,
        timeout: Duration,
    ) -> Result<AccessPermit, MediaAccessError> {
        let owner = self.clone();
        let root = normalize(root)?;
        let queued_root = root.clone();
        let admission = async move {
            loop {
                // Register before checking the registry, so release cannot be lost.
                let changed = owner.changed.notified();
                tokio::pin!(changed);
                changed.as_mut().enable();
                {
                    let mut active =
                        owner
                            .active
                            .lock()
                            .map_err(|error| MediaAccessError::Unavailable {
                                root: root.clone(),
                                message: error.to_string(),
                            })?;
                    let overlaps = active
                        .iter()
                        .any(|path| path.starts_with(&root) || root.starts_with(path));
                    if !overlaps && active.len() < owner.limit {
                        active.push(root.clone());
                        return Ok(AccessPermit {
                            owner: owner.clone(),
                            root,
                        });
                    }
                }
                changed.await;
            }
        };
        tokio::time::timeout(timeout, admission)
            .await
            .unwrap_or_else(|_| {
                Err(MediaAccessError::Busy {
                    root: queued_root,
                    waited: timeout,
                })
            })
    }
}

/// Lexical identity only: canonicalization would itself probe the filesystem.
fn normalize(path: &Path) -> Result<PathBuf, MediaAccessError> {
    let absolute = if path.is_absolute() {
        path.to_owned()
    } else {
        std::env::current_dir()
            .map_err(|error| MediaAccessError::Unavailable {
                root: path.to_owned(),
                message: error.to_string(),
            })?
            .join(path)
    };
    let mut normalized = PathBuf::new();
    for component in absolute.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                normalized.pop();
            }
            other => normalized.push(other.as_os_str()),
        }
    }
    Ok(normalized)
}

/// The OS thread owns this permit until it exits, even if its caller times out.
struct AccessPermit {
    owner: Arc<MediaAccess>,
    root: PathBuf,
}

impl Drop for AccessPermit {
    fn drop(&mut self) {
        // A panic in an operation never occurs while this mutex is held.
        if let Ok(mut active) = self.owner.active.lock() {
            active.retain(|root| root != &self.root);
        }
        self.owner.changed.notify_waiters();
    }
}

/// A root admitted as a directory, only constructible on the filesystem thread.
pub(crate) struct AvailableMediaRoot(PathBuf);

impl AvailableMediaRoot {
    pub(crate) fn path(&self) -> &Path {
        &self.0
    }
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum MediaAccessError {
    #[error("media root '{}' is missing or is not a directory", root.display())]
    Missing { root: PathBuf },
    #[error("media access capacity was unavailable for '{}' for {waited:?}; the root was not inspected", root.display())]
    Busy { root: PathBuf, waited: Duration },
    #[error("media root '{}' did not respond within {waited:?}", root.display())]
    Unresponsive { root: PathBuf, waited: Duration },
    #[error("cannot access media root '{}': {message}", root.display())]
    Unavailable { root: PathBuf, message: String },
}

pub(crate) async fn access<T: Send + 'static>(
    root: PathBuf,
    operation: impl FnOnce(AvailableMediaRoot) -> T + Send + 'static,
) -> Result<T, MediaAccessError> {
    bounded(root, MEDIA_ACCESS_TIMEOUT, ACCESS.clone(), operation).await
}

async fn bounded<T: Send + 'static>(
    root: PathBuf,
    timeout: Duration,
    owner: Arc<MediaAccess>,
    operation: impl FnOnce(AvailableMediaRoot) -> T + Send + 'static,
) -> Result<T, MediaAccessError> {
    let deadline = tokio::time::Instant::now() + timeout;
    let permit = owner.admit(&root, timeout).await?;
    let (send, receive) = oneshot::channel();
    let thread_path = root.clone();
    // Detached threads bound uninterruptible syscalls without delaying runtime shutdown.
    std::thread::Builder::new()
        .name("media-access".into())
        .spawn(move || {
            let _permit = permit;
            let result = match std::fs::metadata(&thread_path) {
                Ok(metadata) if metadata.is_dir() => Ok(operation(AvailableMediaRoot(thread_path))),
                Ok(_) => Err(MediaAccessError::Missing { root: thread_path }),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                    Err(MediaAccessError::Missing { root: thread_path })
                }
                Err(error) => Err(MediaAccessError::Unavailable {
                    root: thread_path,
                    message: error.to_string(),
                }),
            };
            drop(_permit);
            let _ = send.send(result);
        })
        .map_err(|error| MediaAccessError::Unavailable {
            root: root.clone(),
            message: error.to_string(),
        })?;
    match tokio::time::timeout_at(deadline, receive).await {
        Ok(Ok(result)) => result,
        Ok(Err(error)) => Err(MediaAccessError::Unavailable {
            root,
            message: error.to_string(),
        }),
        Err(_) => Err(MediaAccessError::Unresponsive {
            root,
            waited: timeout,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn stalled_root_cannot_consume_capacity_for_an_independent_root() {
        let stalled = tempfile::tempdir().unwrap();
        let healthy = tempfile::tempdir().unwrap();
        let owner = Arc::new(MediaAccess::new(2));
        let (release, wait) = std::sync::mpsc::channel::<()>();
        let (started, entered) = oneshot::channel();
        let lookup = bounded(
            stalled.path().to_owned(),
            Duration::from_millis(100),
            owner.clone(),
            move |_| {
                let _ = started.send(());
                let _ = wait.recv();
            },
        );
        let (result, entered) = tokio::join!(lookup, entered);
        assert!(entered.is_ok());
        assert!(matches!(result, Err(MediaAccessError::Unresponsive { .. })));
        for _ in 0..4 {
            let queued = bounded(
                stalled.path().join("subdir/.."),
                Duration::from_millis(10),
                owner.clone(),
                |_| panic!("root already active"),
            );
            assert!(matches!(queued.await, Err(MediaAccessError::Busy { .. })));
        }
        assert!(
            bounded(
                healthy.path().to_owned(),
                Duration::from_secs(1),
                owner.clone(),
                |root| root.path().is_dir()
            )
            .await
            .unwrap()
        );
        assert_eq!(owner.active.lock().unwrap().len(), 1);
        release.send(()).unwrap();
        // Admission waits for the original thread to release ownership.
        assert!(
            bounded(
                stalled.path().to_owned(),
                Duration::from_secs(1),
                owner,
                |_| true
            )
            .await
            .unwrap()
        );
    }

    #[tokio::test]
    async fn capacity_exhaustion_does_not_claim_the_uninspected_root_is_unresponsive() {
        let owner = Arc::new(MediaAccess::new(1));
        let first = tempfile::tempdir().unwrap();
        let second = tempfile::tempdir().unwrap();
        let _permit = owner
            .admit(first.path(), Duration::from_secs(1))
            .await
            .unwrap();
        assert!(matches!(
            bounded(
                second.path().to_owned(),
                Duration::from_millis(10),
                owner,
                |_| panic!("no capacity")
            )
            .await,
            Err(MediaAccessError::Busy { .. })
        ));
    }

    #[tokio::test]
    async fn available_and_missing_are_distinct() {
        let dir = tempfile::tempdir().unwrap();
        assert!(
            access(dir.path().to_owned(), |root| root.path().is_dir())
                .await
                .unwrap()
        );
        assert!(matches!(
            access(dir.path().join("absent"), |_| ()).await,
            Err(MediaAccessError::Missing { .. })
        ));
    }
}
