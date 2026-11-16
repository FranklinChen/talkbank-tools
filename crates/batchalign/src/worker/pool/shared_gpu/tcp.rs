//! `SharedGpuTcpWorker`: a shared GPU daemon this server connected to over
//! TCP.
//!
//! Everything about the request stream is the shared
//! [`SharedGpuChannel`](super::channel::SharedGpuChannel), the same one the
//! stdio worker uses, liveness included. The difference is ownership: the
//! daemon is not this server's process, so retiring the worker sends
//! `shutdown` and drops the connection, never a signal.

use crate::types::worker_v2::ExecuteResponseV2;
use crate::worker::WorkerPid;
use crate::worker::error::WorkerError;

use super::channel::SharedGpuChannel;

/// A GPU daemon that serves V2 requests concurrently over one TCP connection.
pub(crate) struct SharedGpuTcpWorker {
    channel: SharedGpuChannel<tokio::io::WriteHalf<tokio::net::TcpStream>>,
}

impl SharedGpuTcpWorker {
    /// Consume the actual probed/admitted stream and preserve buffered bytes.
    pub(in crate::worker::pool) fn from_registry(
        worker: crate::worker::registry::AdmittedRegistryWorker,
        capacity: std::num::NonZeroU32,
    ) -> (super::super::WorkerKey, Self) {
        let (key, parts) = crate::worker::tcp_handle::TcpWorkerHandle::promote_registry(worker);
        let channel = SharedGpuChannel::open(
            parts.reader,
            parts.writer,
            parts.info.pid,
            parts.info.task_timeouts,
            capacity.get(),
        );
        (key, Self { channel })
    }
    #[cfg(test)]
    async fn connect(
        info: crate::worker::tcp_handle::TcpWorkerInfo,
        capacity: std::num::NonZeroU32,
    ) -> Result<Self, WorkerError> {
        let stream =
            crate::worker::tcp_handle::connect_within(&format!("{}:{}", info.host, info.port))
                .await?;
        let (reader, writer) = tokio::io::split(stream);
        Ok(Self {
            channel: SharedGpuChannel::open(
                tokio::io::BufReader::new(reader),
                writer,
                info.pid,
                info.task_timeouts,
                capacity.get(),
            ),
        })
    }

    /// Send one typed V2 execute request concurrently.
    pub(crate) async fn execute_v2(
        &self,
        prepared: crate::worker::pool::ReadyRequest,
        preparation: &crate::worker::pool::LoadPreparation,
        registration: crate::worker::pool::job_tracker::TrackerGuard,
    ) -> Result<ExecuteResponseV2, WorkerError> {
        self.channel
            .execute_v2(prepared, preparation, registration)
            .await
    }

    /// Prepare this daemon for `load` (`ensure_task`), returning its proof.
    pub(crate) async fn ensure_task(
        &self,
        load: &crate::worker::pool::TaskLoad,
        preparation: &crate::worker::pool::LoadPreparation,
    ) -> Result<std::sync::Arc<crate::worker::pool::Loaded>, WorkerError> {
        self.channel.ensure_task(load, preparation).await
    }

    /// Query this exact TCP worker's live capability and engine identity.
    pub(crate) async fn capabilities(
        &self,
    ) -> Result<crate::worker::WorkerCapabilities, WorkerError> {
        self.channel.capabilities().await
    }

    /// Whether the connection can still take requests: refused once the
    /// daemon's stream has closed or shutdown has begun.
    pub(crate) fn check_available(&self) -> Result<(), WorkerError> {
        self.channel.check_available()
    }

    /// Ask the daemon to end this connection and stop reading it, for
    /// `why`. The daemon process itself is managed outside this server.
    pub(crate) async fn shutdown(&self, why: super::Retirement) {
        // The acknowledgement is not awaited: nothing here waits on the
        // daemon, which keeps running for other connections.
        let _ack = self.channel.begin_shutdown(why).await;
        self.channel.close(why);
    }

    /// The worker process ID.
    pub(crate) fn pid(&self) -> WorkerPid {
        self.channel.pid()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    use crate::api::{LanguageCode3, WorkerLanguage};
    use crate::types::worker_v2::TaskTimeoutOverrides;
    use crate::worker::WorkerProfile;
    use crate::worker::tcp_handle::TcpWorkerInfo;

    /// Guards: The actual shared reader observes peer EOF and refuses later use of that same daemon connection.
    /// Kind: contract
    /// Not a type: A peer can close the real socket after static admission. Kernel EOF and the background reader's availability projection remain native effects after ownership is fixed.
    /// State: One private loopback listener and its accepted peer exercise the shared channel with no model or daemon process.
    /// Concurrency: The fixture owns both socket endpoints and awaits acceptance before dropping the peer. A bounded wait observes the reader's EOF transition, with no shared ports or mutable environment.
    #[tokio::test]
    async fn a_closed_daemon_connection_reports_itself_unavailable() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind a local port");
        let port = listener.local_addr().expect("local address").port();
        let accepted = tokio::spawn(async move { listener.accept().await });
        let worker = SharedGpuTcpWorker::connect(
            TcpWorkerInfo {
                host: "127.0.0.1".into(),
                port,
                profile: WorkerProfile::Gpu,
                lang: WorkerLanguage::from(LanguageCode3::eng()),
                engine_overrides: String::new(),
                pid: WorkerPid(4242),
                task_timeouts: TaskTimeoutOverrides::NONE,
            },
            std::num::NonZeroU32::new(2).unwrap(),
        )
        .await
        .expect("connect to the local listener");
        let (daemon_side, _) = accepted
            .await
            .expect("accept task")
            .expect("accept the connection");
        worker
            .check_available()
            .expect("an open connection is available");

        drop(daemon_side);
        // The reader observes EOF asynchronously; wait for it, bounded.
        tokio::time::timeout(Duration::from_secs(5), async {
            while worker.check_available().is_ok() {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("the reader must observe the closed connection");
        assert!(matches!(
            worker.check_available(),
            Err(WorkerError::ProcessExited { .. })
        ));
    }
}
