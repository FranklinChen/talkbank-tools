//! Startup admission shared by real workers and their native fixtures.

use serde::Deserialize;

use crate::worker::WorkerPid;
use crate::worker::error::WorkerError;
use crate::worker::runtime_identity::WorkerRuntimeIdentity;

/// A decoded affirmative handshake retaining the worker's runtime evidence.
#[derive(Debug, Deserialize)]
#[serde(try_from = "ReadyWire")]
pub(in crate::worker::handle) struct ReadySignal {
    pid: WorkerPid,
    runtime: WorkerRuntimeIdentity,
    transport: ReadyTransport,
}

#[derive(Deserialize)]
struct ReadyWire {
    ready: bool,
    pid: u32,
    #[serde(flatten)]
    transport: ReadyTransport,
    runtime: WorkerRuntimeIdentity,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "transport", rename_all = "snake_case")]
enum ReadyTransport {
    Stdio,
    Tcp { host: String, port: u16 },
}

impl TryFrom<ReadyWire> for ReadySignal {
    type Error = &'static str;

    fn try_from(wire: ReadyWire) -> Result<Self, Self::Error> {
        if !wire.ready {
            return Err("worker emitted ready line with ready=false");
        }
        Ok(Self {
            pid: WorkerPid(wire.pid),
            runtime: wire.runtime,
            transport: wire.transport,
        })
    }
}

/// Startup evidence admitted specifically for a stdio handle.
pub(in crate::worker::handle) struct StdioReady {
    pid: WorkerPid,
    runtime: WorkerRuntimeIdentity,
}

/// Startup evidence admitted specifically for a TCP daemon.
pub(in crate::worker::handle) struct TcpReady {
    pid: WorkerPid,
    runtime: WorkerRuntimeIdentity,
    host: String,
    port: u16,
}

impl ReadySignal {
    pub(in crate::worker::handle) fn into_stdio(self) -> Result<StdioReady, WorkerError> {
        match self.transport {
            ReadyTransport::Stdio => Ok(StdioReady {
                pid: self.pid,
                runtime: self.runtime,
            }),
            ReadyTransport::Tcp { .. } => Err(WorkerError::ReadyParseFailed(
                "a worker spawned for stdio reported a TCP ready line".into(),
            )),
        }
    }

    pub(in crate::worker::handle) fn into_tcp(self) -> Result<TcpReady, WorkerError> {
        match self.transport {
            ReadyTransport::Tcp { host, port } => Ok(TcpReady {
                pid: self.pid,
                runtime: self.runtime,
                host,
                port,
            }),
            ReadyTransport::Stdio => Err(WorkerError::ReadyParseFailed(
                "a worker spawned as a TCP daemon reported a stdio ready line".into(),
            )),
        }
    }
}

impl StdioReady {
    pub(in crate::worker::handle) fn into_parts(self) -> (WorkerPid, WorkerRuntimeIdentity) {
        (self.pid, self.runtime)
    }
}

impl TcpReady {
    pub(in crate::worker::handle) fn into_parts(
        self,
    ) -> (WorkerPid, WorkerRuntimeIdentity, String, u16) {
        (self.pid, self.runtime, self.host, self.port)
    }
}
