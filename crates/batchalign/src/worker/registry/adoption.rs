//! A pool's adoption claim lives with its actual daemon connection.
use super::{RegistryEntry, admission::PreparedEntry};
use crate::worker::error::WorkerError;
use crate::worker::pool::{WorkerKey, WorkerPool};
use crate::worker::tcp_handle::{TcpWorkerHandle, TcpWorkerInfo};
use crate::worker::{WorkerCapabilities, WorkerPid};
use std::collections::HashSet;
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct DaemonIdentity {
    host: String,
    port: std::num::NonZeroU16,
    pid: crate::worker::process_id::NativePid,
}
impl DaemonIdentity {
    fn of(entry: &RegistryEntry) -> Result<Self, WorkerError> {
        Ok(Self {
            host: entry.host.clone(),
            port: std::num::NonZeroU16::new(entry.port).ok_or_else(|| {
                WorkerError::Protocol("registry worker port must be positive".into())
            })?,
            pid: crate::worker::process_id::NativePid::admit(entry.pid).ok_or_else(|| {
                WorkerError::Protocol(
                    "registry worker PID is not a positive native process ID".into(),
                )
            })?,
        })
    }
}
#[derive(Clone, Default)]
pub(in crate::worker) struct RegistryAdmissions(Arc<Mutex<HashSet<DaemonIdentity>>>);
/// Constructors stay with admission. Readers cannot detach or reconstruct a claim.
pub(in crate::worker) struct ConnectionOwnership {
    _origin: Origin,
}
enum Origin {
    Independent,
    Registry { _lease: RegistryLease },
}
struct RegistryLease {
    identity: DaemonIdentity,
    owners: Arc<Mutex<HashSet<DaemonIdentity>>>,
}
impl Drop for RegistryLease {
    fn drop(&mut self) {
        self.owners
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(&self.identity);
    }
}
impl ConnectionOwnership {
    pub(in crate::worker) fn independent() -> Self {
        Self {
            _origin: Origin::Independent,
        }
    }
}
/// A small, exclusive claim on the original source before key preparation.
pub(super) struct ClaimedRegistryEntry<'source> {
    source: &'source RegistryEntry,
    profile: crate::worker::WorkerProfile,
    language: crate::api::WorkerLanguage,
    ownership: ConnectionOwnership,
}

pub(super) enum ClaimDisposition<'source> {
    Claimed(ClaimedRegistryEntry<'source>),
    AlreadyClaimed,
}
impl RegistryAdmissions {
    pub(super) fn claim<'source>(
        &self,
        prepared: PreparedEntry<'source>,
    ) -> Result<ClaimDisposition<'source>, WorkerError> {
        let (source, profile, language) = prepared.into_parts();
        let identity = DaemonIdentity::of(source)?;
        let mut owners = self
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !owners.insert(identity.clone()) {
            return Ok(ClaimDisposition::AlreadyClaimed);
        }
        Ok(ClaimDisposition::Claimed(ClaimedRegistryEntry {
            source,
            profile,
            language,
            ownership: ConnectionOwnership {
                _origin: Origin::Registry {
                    _lease: RegistryLease {
                        identity,
                        owners: self.0.clone(),
                    },
                },
            },
        }))
    }
}

/// A normalized key and connection parameters derived from the claimed source.
pub(super) struct PreparedRegistryConnection {
    source: RegistryEntry,
    info: TcpWorkerInfo,
    key: WorkerKey,
    ownership: ConnectionOwnership,
}

impl ClaimedRegistryEntry<'_> {
    pub(super) fn prepare(
        self,
        timeouts: crate::types::worker_v2::TaskTimeoutOverrides,
    ) -> Result<PreparedRegistryConnection, WorkerError> {
        let Self {
            source,
            profile,
            language,
            ownership,
        } = self;
        let key = WorkerKey::from_registry_json(
            crate::worker::WorkerTarget::profile(profile),
            language.clone(),
            &source.engine_overrides,
        )
        .map_err(|error| {
            WorkerError::Protocol(format!(
                "registry engine overrides could not be admitted: {error}"
            ))
        })?;
        Ok(PreparedRegistryConnection {
            source: source.clone(),
            info: TcpWorkerInfo {
                host: source.host.clone(),
                port: source.port,
                profile,
                lang: language,
                engine_overrides: source.engine_overrides.clone(),
                pid: WorkerPid(source.pid),
                task_timeouts: timeouts,
            },
            key,
            ownership,
        })
    }
}
impl PreparedRegistryConnection {
    pub(super) async fn connect(
        self,
    ) -> Result<ConnectedRegistryWorker, Box<RejectedRegistryWorker>> {
        let Self {
            source,
            info,
            key,
            ownership,
        } = self;
        match TcpWorkerHandle::connect_owned(info, ownership).await {
            Ok(handle) => Ok(ConnectedRegistryWorker {
                source,
                key,
                handle,
            }),
            Err((error, ownership)) => Err(Box::new(RejectedRegistryWorker::new(
                source, error, ownership,
            ))),
        }
    }
}

/// A failed probe keeps its original claim until stale cleanup finishes.
pub(super) struct RejectedRegistryWorker {
    source: RegistryEntry,
    error: WorkerError,
    _ownership: ConnectionOwnership,
}
impl RejectedRegistryWorker {
    fn new(source: RegistryEntry, error: WorkerError, ownership: ConnectionOwnership) -> Self {
        Self {
            source,
            error,
            _ownership: ownership,
        }
    }

    pub(super) fn source(&self) -> &RegistryEntry {
        &self.source
    }

    pub(super) fn error(&self) -> &WorkerError {
        &self.error
    }
}

pub(super) struct ConnectedRegistryWorker {
    source: RegistryEntry,
    key: WorkerKey,
    handle: TcpWorkerHandle,
}
impl ConnectedRegistryWorker {
    pub(super) async fn check(
        mut self,
    ) -> Result<ProbedRegistryWorker, Box<RejectedRegistryWorker>> {
        if let Err(error) = self.handle.health_check().await {
            return Err(Box::new(RejectedRegistryWorker::new(
                self.source,
                error,
                self.handle.close(),
            )));
        }
        Ok(ProbedRegistryWorker {
            source: self.source,
            key: self.key,
            handle: self.handle,
        })
    }
}
/// Only an actual successful health exchange can create this owner.
pub(super) struct ProbedRegistryWorker {
    source: RegistryEntry,
    key: WorkerKey,
    handle: TcpWorkerHandle,
}
/// A failed incomplete exchange cannot produce a report-ready live owner.
pub(in crate::worker) struct ReportReadyRegistryWorker {
    probed: ProbedRegistryWorker,
    report: ReportEvidence,
}
enum ReportEvidence {
    Observed(WorkerCapabilities),
    Deferred,
    ReportedFailure(WorkerError),
}
impl ProbedRegistryWorker {
    pub(super) async fn observe_capabilities(
        mut self,
    ) -> Result<ReportReadyRegistryWorker, WorkerError> {
        let report = match self.handle.capabilities().await {
            Ok(report) => ReportEvidence::Observed(report),
            Err(error) => match error.worker_after_failure() {
                crate::worker::error::WorkerAfterFailure::Reusable => {
                    ReportEvidence::ReportedFailure(error)
                }
                crate::worker::error::WorkerAfterFailure::Retire => return Err(error),
            },
        };
        Ok(ReportReadyRegistryWorker {
            probed: self,
            report,
        })
    }
    pub(super) fn defer_report(self) -> ReportReadyRegistryWorker {
        ReportReadyRegistryWorker {
            probed: self,
            report: ReportEvidence::Deferred,
        }
    }
}
impl ReportReadyRegistryWorker {
    pub(super) fn observed_report(&self) -> bool {
        matches!(&self.report, ReportEvidence::Observed(_))
    }
    pub(in crate::worker) fn admit(
        self,
        pool: &WorkerPool,
    ) -> Result<AdmittedRegistryWorker, WorkerError> {
        let Self { probed, report } = self;
        let ProbedRegistryWorker {
            source,
            key,
            handle,
        } = probed;
        match report {
            ReportEvidence::Observed(report) => {
                pool.record_capabilities(&key, report)?;
            }
            ReportEvidence::ReportedFailure(error) => {
                tracing::warn!(pid=source.pid,%error,"Registry worker reported a capability failure")
            }
            ReportEvidence::Deferred => {}
        }
        Ok(AdmittedRegistryWorker {
            source,
            key,
            handle,
        })
    }
}
/// The report was admitted under the key derived from this connection's source.
pub(in crate::worker) struct AdmittedRegistryWorker {
    source: RegistryEntry,
    key: WorkerKey,
    handle: TcpWorkerHandle,
}
impl AdmittedRegistryWorker {
    pub(in crate::worker) fn key(&self) -> &WorkerKey {
        &self.key
    }
    pub(in crate::worker) fn info(&self) -> &TcpWorkerInfo {
        self.handle.info()
    }
    pub(in crate::worker) fn into_sequential(self) -> (WorkerKey, TcpWorkerHandle) {
        (self.key, self.handle)
    }
    pub(in crate::worker) fn source(&self) -> &RegistryEntry {
        &self.source
    }
}
