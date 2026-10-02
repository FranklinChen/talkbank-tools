//! `SharedGpuTcpWorker`: a shared GPU daemon this server connected to over
//! TCP.
//!
//! Everything about the request stream is the shared
//! [`SharedGpuChannel`](super::channel::SharedGpuChannel), the same one the
//! stdio worker uses, liveness included. The difference is ownership: the
//! daemon is not this server's process, so retiring the worker sends
//! `shutdown` and drops the connection, never a signal.

use tokio::io::BufReader;

use crate::types::worker_v2::{ExecuteRequestV2, ExecuteResponseV2};
use crate::worker::WorkerPid;
use crate::worker::error::WorkerError;

use super::channel::SharedGpuChannel;

/// A GPU daemon that serves V2 requests concurrently over one TCP connection.
pub(crate) struct SharedGpuTcpWorker {
    channel: SharedGpuChannel<tokio::io::WriteHalf<tokio::net::TcpStream>>,
}

impl SharedGpuTcpWorker {
    /// Connect to a TCP GPU worker and start routing its replies.
    pub(crate) async fn connect(
        info: crate::worker::tcp_handle::TcpWorkerInfo,
    ) -> Result<Self, WorkerError> {
        let addr = format!("{}:{}", info.host, info.port);
        let stream = crate::worker::tcp_handle::connect_within(&addr).await?;

        let (read_half, write_half) = tokio::io::split(stream);
        Ok(Self {
            channel: SharedGpuChannel::open(
                BufReader::new(read_half),
                write_half,
                info.pid,
                info.task_timeouts,
                info.gpu_thread_pool_size,
            ),
        })
    }

    /// Send one typed V2 execute request concurrently.
    pub(crate) async fn execute_v2(
        &self,
        request: &ExecuteRequestV2,
    ) -> Result<ExecuteResponseV2, WorkerError> {
        self.channel.execute_v2(request).await
    }

    /// Load one task in a lazy TCP worker.
    pub(crate) async fn ensure_task(
        &self,
        task: crate::worker::InferTask,
        engine_overrides: Option<&std::collections::BTreeMap<String, String>>,
        timeout: crate::api::PositiveSeconds,
    ) -> Result<(), WorkerError> {
        self.channel
            .ensure_task(task, engine_overrides, timeout)
            .await
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

    /// A daemon connection whose stream has ended reports itself unavailable,
    /// the observation the pool uses to drop it and dispatch elsewhere rather
    /// than leave every request to wait out its timeout on a dead connection.
    #[tokio::test]
    async fn a_closed_daemon_connection_reports_itself_unavailable() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind a local port");
        let port = listener.local_addr().expect("local address").port();
        let accepted = tokio::spawn(async move { listener.accept().await });
        let worker = SharedGpuTcpWorker::connect(TcpWorkerInfo {
            host: "127.0.0.1".into(),
            port,
            profile: WorkerProfile::Gpu,
            lang: WorkerLanguage::from(LanguageCode3::eng()),
            engine_overrides: String::new(),
            pid: WorkerPid(4242),
            task_timeouts: TaskTimeoutOverrides::NONE,
            gpu_thread_pool_size: 2,
        })
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
