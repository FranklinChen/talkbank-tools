//! Read helper: consume `progress_v2` preamble events while waiting for
//! the final IPC response.
//!
//! The Python worker emits `{"op": "progress_v2", ...}` lines on stdout
//! during long-running cache-relevant operations: Stanza catalog
//! download, Stanza per-language pack download, `HuggingFace` model
//! download (Whisper / pyannote / fasttext / transformers), torch
//! checkpoint warm-up, and similar. These events arrive interleaved
//! with the actual request/response IPC.
//!
//! Every supervisor-side read site that follows a request whose
//! handler can trigger a model load must tolerate `progress_v2`
//! preamble: otherwise the very first job that triggers the
//! download fails with
//! `unexpected response for {op}: ProgressV2 { ... }` and the
//! supervisor aborts. The 2026-05-06 morphotag incident
//! (`read_ready_line` fix) and the 2026-05-09 capabilities-probe
//! incident were two instances of the same bug class in different
//! handshake phases.
//!
//! This module factors that loop discipline into a single helper so
//! every cache-relevant read site shares one tested implementation
//! rather than each maintaining its own match-and-reject.

use tokio::sync::mpsc;

use crate::types::worker_v2::ProgressEventV2;
use crate::worker::RunningWait;
use crate::worker::error::WorkerError;

use super::WorkerHandle;
use super::control::{ControlCompletion, ControlFrame, ExpectedReply};
use super::protocol::WorkerResponse;

/// A source of worker responses: one complete response per call, by the one
/// line rule ([`super::protocol::WireLine`]), with the source's own account
/// of a pipe that broke or closed.
///
/// The stdio worker ([`WorkerHandle`]), the TCP worker
/// (`crate::worker::TcpWorkerHandle`) and, in tests, a bare line reader
/// implement it, so [`read_skipping_progress`] is the one progress loop all
/// three run. There used to be three copies of that loop, each changed by
/// hand when the wait gained progress-driven re-arming.
pub trait ResponseSource {
    /// The next complete response.
    async fn read_response(&mut self) -> Result<WorkerResponse, WorkerError>;
}

impl ResponseSource for WorkerHandle {
    async fn read_response(&mut self) -> Result<WorkerResponse, WorkerError> {
        // The inherent reader, which drains stderr into `ProcessExited` on a
        // broken or closed pipe.
        WorkerHandle::read_response(self).await
    }
}

/// Read responses from `source` until a non-`ProgressV2` one arrives, or
/// the operation's own wait ends.
///
/// Each `ProgressV2` event is shown to the wait (a loading wait re-arms on an
/// advance, a budget on nothing; [`RunningWait::observe`]) and optionally
/// forwarded to `progress_tx`, dropped if the channel is closed or full:
/// progress is observability, not flow control, and a slow consumer must not
/// stall the worker. The consumed operation admits the terminal before a
/// completion proof can authorize worker reuse. A foreign terminal remains an
/// incomplete exchange.
pub async fn read_skipping_progress<S: ResponseSource, E: ExpectedReply>(
    source: &mut S,
    expected: E,
    progress_tx: Option<&mpsc::Sender<ProgressEventV2>>,
) -> Result<ControlCompletion<E::Reply>, WorkerError> {
    let wait = expected.wait();
    let loading = crate::worker::pool::checkout_wait::ObservedModelLoad::for_wait(&wait);
    let mut running = RunningWait::start(wait);
    loop {
        let response = running.within(source.read_response()).await??;
        match ControlFrame::from(response) {
            ControlFrame::Progress(event) => {
                running.observe(&event);
                if let (Some(loading), ProgressEventV2::Load(load)) = (&loading, &event) {
                    loading.report(load);
                }
                if let Some(tx) = progress_tx {
                    let _ = tx.try_send(event);
                }
            }
            ControlFrame::Terminal(terminal) => return expected.admit(terminal),
        }
    }
}

/// A bare line reader as a [`ResponseSource`], for tests: the one line rule,
/// with an end of stream or a read error reported as a protocol error (there
/// is no child process to drain).
#[cfg(test)]
struct LineSource<'a, R>(&'a mut R);

#[cfg(test)]
impl<R: tokio::io::AsyncBufRead + Unpin> ResponseSource for LineSource<'_, R> {
    async fn read_response(&mut self) -> Result<WorkerResponse, WorkerError> {
        use serde::Deserialize;
        use tokio::io::AsyncBufReadExt;
        let mut noise = super::protocol::NoiseRun::default();
        loop {
            let mut line = String::new();
            let bytes = self.0.read_line(&mut line).await.map_err(|io_err| {
                WorkerError::Protocol(format!("I/O error reading response: {io_err}"))
            })?;
            if bytes == 0 {
                return Err(WorkerError::Protocol(
                    "worker closed stdout before delivering a final response".into(),
                ));
            }
            match super::protocol::WireLine::classify(&line) {
                super::protocol::WireLine::Blank => {}
                super::protocol::WireLine::Noise => noise
                    .noise(&line)
                    .map_err(super::protocol::NoiseLimit::into_worker_error)?,
                super::protocol::WireLine::Message(message) => {
                    return WorkerResponse::deserialize(&message).map_err(|e| {
                        WorkerError::Protocol(format!(
                            "failed to decode worker response: {e} (line: {})",
                            super::protocol::excerpt(line.trim())
                        ))
                    });
                }
            }
        }
    }
}

#[cfg(test)]
mod progress_preamble_tests;
