//! Application state (server-specific composition root).
//!
//! Capability discovery and validation logic lives in [`crate::capability`].
//! This module re-exports those types for backward compatibility.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use crate::capability::WorkerCapabilitySnapshot;
use crate::config::ServerConfig;
use crate::media::MediaResolver;
use crate::runtime_supervisor::{ShutdownError, ShutdownSummary};
use crate::server_backend::ServerBackend;
use crate::worker::InferTask;
use crate::worker::pool::WorkerPool;

// ---------------------------------------------------------------------------
// Application state
// ---------------------------------------------------------------------------

/// Shared coordination handles for the server control plane.
///
/// These are the mutable runtime boundaries that routes and background tasks
/// must collaborate through instead of open-coding separate store, queue,
/// runtime, and broadcast dependencies.
pub(crate) struct AppControlPlane {
    /// Route-facing backend for queued-job orchestration and persisted state.
    pub backend: Arc<dyn ServerBackend>,
}

/// Worker-facing runtime dependencies and the capability view resolved when
/// the app was built.
pub(crate) struct WorkerSubsystem {
    /// Pool of Python worker processes for ML inference.
    pub pool: Arc<WorkerPool>,
    /// Released commands and the infer tasks behind them, derived together
    /// from the pool's admitted report (one copy, read through accessors).
    pub capabilities: WorkerCapabilitySnapshot,
}

/// Filesystem roots owned by the server process.
pub(crate) struct AppPaths {
    /// Root directory for per-job staging folders.
    pub jobs_dir: PathBuf,
    /// On-disk dashboard SPA root when runtime assets override embedded files.
    pub dashboard_dir: Option<PathBuf>,
}

/// Immutable environment configuration shared by HTTP handlers.
pub(crate) struct AppEnvironment {
    /// Server configuration loaded from the runtime-owned `server.yaml`.
    pub config: ServerConfig,
    /// The admitted `paths_mode_roots`, resolved once at startup.
    pub allowed_roots: batchalign_types::paths::AllowedRoots,
    /// This server's same-user key, published in its handshake: a request
    /// proved under it is from this server's own account (see
    /// [`crate::same_user_key`]).
    pub same_user_key: crate::same_user_key::ServerKey,
    /// Nonces issued for same-user proofs, each good once.
    pub same_user_nonces: crate::same_user_key::NonceStore,
    /// Media resolver with a cached view of configured media roots.
    pub media: MediaResolver,
    /// Server-managed filesystem roots.
    pub paths: AppPaths,
}

impl AppEnvironment {
    /// The authority a request's paths are admitted under: the user's own
    /// process for a valid same-user proof from this machine, the configured
    /// folders for a request with none. A proof that does not hold is
    /// refused rather than downgraded.
    pub(crate) fn authority_for(
        &self,
        headers: &axum::http::HeaderMap,
        peer: std::net::SocketAddr,
        request: &crate::same_user_key::RequestBinding<'_>,
    ) -> Result<batchalign_types::paths::AllowedRoots, crate::error::ServerError> {
        use crate::same_user_key::{HEADER, KeyRefusal, PeerLocality, Requester};
        let presented = match headers.get(HEADER) {
            None => None,
            Some(value) => Some(value.to_str().map_err(|_| KeyRefusal::Malformed)?),
        };
        Ok(
            match self.same_user_key.requester(
                presented,
                PeerLocality::of(peer),
                request,
                &self.same_user_nonces,
            )? {
                Requester::SameUser => batchalign_types::paths::AllowedRoots::same_user(),
                Requester::Unidentified => self.allowed_roots.clone(),
            },
        )
    }
}

/// Build and version identity surfaced to clients.
pub(crate) struct AppBuildInfo {
    /// Crate version string from `Cargo.toml`.
    pub version: String,
    /// Rebuild fingerprint used for daemon restart detection.
    pub build_hash: String,
}

/// Shared application state, available to all route handlers via `State<Arc<AppState>>`.
///
/// The root state intentionally stays shallow. Wide mutable infrastructure
/// fields belong inside named sub-aggregates so routes depend on the specific
/// boundary they are crossing: control plane, worker subsystem, environment,
/// or build identity.
pub struct AppState {
    /// Shared control-plane coordination handles.
    pub(crate) control: AppControlPlane,
    /// Worker pool handle plus the startup capability profile.
    pub(crate) workers: WorkerSubsystem,
    /// Immutable environment and filesystem configuration.
    pub(crate) environment: AppEnvironment,
    /// Version/build identity reported to clients.
    pub(crate) build: AppBuildInfo,
}

impl AppState {
    /// Return the command capability set advertised by the worker subsystem.
    pub fn capabilities(&self) -> &[crate::api::ReleasedCommand] {
        self.workers.capabilities.commands()
    }

    /// Return the infer-task set advertised by the worker subsystem.
    pub fn infer_tasks(&self) -> &[InferTask] {
        self.workers.capabilities.infer_tasks()
    }

    /// Interrupt queued/running jobs and stop tracked background tasks for fixture reuse.
    ///
    /// Reused-worker fixtures call this before dropping one isolated app
    /// instance so no job task or queue loop keeps running against the shared
    /// worker pool after the control plane has been torn down.
    pub async fn shutdown_for_reuse(
        &self,
        timeout: Duration,
    ) -> Result<ShutdownSummary, ShutdownError> {
        let _ = self.control.backend.interrupt_all_for_shutdown().await;
        self.control.backend.shutdown_runtime(timeout).await
    }
}

// Tests live in crate::capability::tests.
