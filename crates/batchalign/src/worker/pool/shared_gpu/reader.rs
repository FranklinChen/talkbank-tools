//! Generic response reader loop shared between stdio and TCP GPU workers.
//!
//! Reads JSON-lines from any `AsyncBufRead`, routing V2 responses by
//! `request_id` to pending oneshot senders, and non-V2 responses
//! (health, capabilities, shutdown, error) to a control channel.

use std::collections::HashMap;
use std::sync::Arc;

use serde_json::Value;
use tokio::sync::oneshot;
use tracing::{debug, error, warn};

use tokio::io::AsyncBufReadExt;

use crate::api::DurationSeconds;
use crate::types::worker_v2::{
    ExecuteOutcomeV2, ExecuteResponseV2, ProtocolErrorCodeV2, WorkerRequestIdV2,
};
use crate::worker::WorkerPid;

use super::WorkerControlResponse;
use super::envelopes::{
    CapabilitiesResponseEnvelope, EnsureTaskResponseEnvelope, ExecuteResponseV2Envelope,
    HealthResponseEnvelope,
};

/// Generic reader loop that works with any `AsyncBufRead` — shared between
/// stdio ([`super::SharedGpuWorker`]) and TCP ([`super::SharedGpuTcpWorker`]).
pub(crate) async fn reader_loop_generic<R: tokio::io::AsyncBufRead + Unpin>(
    reader: &mut R,
    pending: Arc<std::sync::Mutex<HashMap<String, oneshot::Sender<ExecuteResponseV2>>>>,
    control: Arc<tokio::sync::Mutex<Option<oneshot::Sender<WorkerControlResponse>>>>,
    pid: WorkerPid,
) {
    let mut line = String::new();
    loop {
        line.clear();
        match reader.read_line(&mut line).await {
            Ok(0) => {
                debug!(pid = %pid, "GPU worker stream closed (EOF)");
                let mut pending = super::super::lock_recovered(&pending);
                for (id, tx) in pending.drain() {
                    debug!(pid = %pid, request_id = %id, "Failing pending request (worker stream closed)");
                    drop(tx);
                }
                break;
            }
            Ok(_) => {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }

                let parsed: Value = match serde_json::from_str(trimmed) {
                    Ok(v) => v,
                    Err(e) => {
                        warn!(
                            pid = %pid,
                            line = trimmed,
                            error = %e,
                            "GPU worker: ignoring non-JSON line"
                        );
                        continue;
                    }
                };

                let op = parsed.get("op").and_then(|v| v.as_str()).unwrap_or("");

                match op {
                    "execute_v2" => {
                        match serde_json::from_value::<ExecuteResponseV2Envelope>(parsed.clone()) {
                            Ok(envelope) => {
                                let request_id = envelope.response.request_id.to_string();
                                let mut pending = super::super::lock_recovered(&pending);
                                if let Some(tx) = pending.remove(&request_id) {
                                    let _ = tx.send(envelope.response);
                                } else {
                                    warn!(
                                        pid = %pid,
                                        request_id = %request_id,
                                        "GPU worker: orphaned execute_v2 response"
                                    );
                                }
                            }
                            Err(e) => {
                                error!(
                                    pid = %pid,
                                    error = %e,
                                    "GPU worker: failed to parse execute_v2 response"
                                );
                            }
                        }
                    }
                    "health" => {
                        if let Ok(envelope) =
                            serde_json::from_value::<HealthResponseEnvelope>(parsed)
                        {
                            let mut ctrl = control.lock().await;
                            if let Some(tx) = ctrl.take() {
                                let _ = tx.send(WorkerControlResponse::Health(envelope.response));
                            }
                        }
                    }
                    "capabilities" => {
                        if let Ok(envelope) =
                            serde_json::from_value::<CapabilitiesResponseEnvelope>(parsed)
                        {
                            let mut ctrl = control.lock().await;
                            if let Some(tx) = ctrl.take() {
                                let _ =
                                    tx.send(WorkerControlResponse::Capabilities(envelope.response));
                            }
                        }
                    }
                    "ensure_task" => {
                        if let Ok(envelope) =
                            serde_json::from_value::<EnsureTaskResponseEnvelope>(parsed)
                        {
                            let mut ctrl = control.lock().await;
                            if let Some(tx) = ctrl.take() {
                                let _ =
                                    tx.send(WorkerControlResponse::EnsureTask(envelope.response));
                            }
                        }
                    }
                    "shutdown" => {
                        let mut ctrl = control.lock().await;
                        if let Some(tx) = ctrl.take() {
                            let _ = tx.send(WorkerControlResponse::Shutdown);
                        }
                    }
                    "error" => {
                        let error_msg = parsed
                            .get("error")
                            .and_then(|v| v.as_str())
                            .unwrap_or("unknown error")
                            .to_string();
                        // If the worker tagged the error with a
                        // `request_id`, route it to the matching V2 pending
                        // oneshot as a synthetic `ExecuteResponseV2` with
                        // `outcome: Error{...}`. Without this, the V2
                        // dispatch sits on its per-request timeout (default
                        // 180s for audio tasks) while the worker has
                        // ALREADY surfaced the error — exactly the failure
                        // mode that hid the 2026-05-27 HK_QWEN schema-drift
                        // bug for ~12 hours. See
                        // `crates/batchalign-pyo3/src/worker_protocol.rs::
                        // error_payload_with_request_id` for the Python
                        // side.
                        let request_id = parsed
                            .get("request_id")
                            .and_then(|v| v.as_str())
                            .map(str::to_owned);
                        if let Some(request_id) = request_id {
                            let routed = {
                                let mut pending = super::super::lock_recovered(&pending);
                                pending.remove(&request_id)
                            };
                            if let Some(tx) = routed {
                                let response = ExecuteResponseV2 {
                                    request_id: WorkerRequestIdV2::from(request_id.clone()),
                                    outcome: ExecuteOutcomeV2::Error {
                                        code: ProtocolErrorCodeV2::InvalidPayload,
                                        message: error_msg.clone(),
                                    },
                                    result: None,
                                    elapsed_s: DurationSeconds(0.0),
                                };
                                debug!(
                                    pid = %pid,
                                    request_id = %request_id,
                                    error = %error_msg,
                                    "GPU worker: routing tagged error to pending V2 dispatch"
                                );
                                let _ = tx.send(response);
                                continue;
                            }
                            warn!(
                                pid = %pid,
                                request_id = %request_id,
                                error = %error_msg,
                                "GPU worker: tagged error has no matching pending dispatch \
                                 (worker may have processed it asynchronously)"
                            );
                        }
                        // Untagged error envelopes still flow through the
                        // control channel for the sequential ops
                        // (health / capabilities / ensure_task) that
                        // expect them there.
                        let mut ctrl = control.lock().await;
                        if let Some(tx) = ctrl.take() {
                            let _ = tx.send(WorkerControlResponse::Error(error_msg));
                        }
                    }
                    _ => {
                        warn!(
                            pid = %pid,
                            op = op,
                            "GPU worker: unexpected response op"
                        );
                    }
                }
            }
            Err(e) => {
                error!(pid = %pid, error = %e, "GPU worker: stream read error");
                // Explicitly fail all pending requests — same as the EOF
                // path. Without this, pending oneshot senders are implicitly
                // dropped when the task exits, causing receivers to see
                // "channel closed" with no useful error context.
                let mut pending = super::super::lock_recovered(&pending);
                let n = pending.len();
                for (id, tx) in pending.drain() {
                    debug!(pid = %pid, request_id = %id, "Failing pending request (I/O error)");
                    drop(tx);
                }
                if n > 0 {
                    error!(
                        pid = %pid,
                        failed_requests = n,
                        error = %e,
                        "GPU worker crashed — failed {n} pending requests"
                    );
                }
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::BufReader;

    /// `op=error` envelopes tagged with `request_id` must fail the matching
    /// V2 dispatch's pending oneshot immediately, rather than be silently
    /// routed to the empty control channel and leave the dispatch sitting
    /// on its per-request timeout. Regression guard for the 2026-05-27
    /// HK_QWEN schema-drift hang.
    #[tokio::test]
    async fn tagged_error_envelope_fails_pending_v2_dispatch() {
        let pid = WorkerPid(12345);
        let pending = Arc::new(std::sync::Mutex::new(HashMap::<
            String,
            oneshot::Sender<ExecuteResponseV2>,
        >::new()));
        let control = Arc::new(tokio::sync::Mutex::new(
            None::<oneshot::Sender<WorkerControlResponse>>,
        ));

        let (tx, rx) = oneshot::channel();
        let request_id = "asr-v2-request-77";
        super::super::super::lock_recovered(&pending).insert(request_id.into(), tx);

        let envelope = format!(
            "{{\"op\":\"error\",\"error\":\"invalid execute_v2 request: ValidationError\",\
             \"request_id\":\"{request_id}\"}}\n",
        );
        let mut reader = BufReader::new(envelope.as_bytes());

        reader_loop_generic(&mut reader, pending.clone(), control.clone(), pid).await;

        let response = rx
            .await
            .expect("pending oneshot must resolve when error is routed");
        assert_eq!(response.request_id.as_ref(), request_id);
        match response.outcome {
            ExecuteOutcomeV2::Error { code, message } => {
                assert_eq!(code, ProtocolErrorCodeV2::InvalidPayload);
                assert!(
                    message.contains("ValidationError"),
                    "expected worker error message to propagate, got: {message}"
                );
            }
            ExecuteOutcomeV2::Success => {
                panic!("tagged error envelope must produce ExecuteOutcomeV2::Error")
            }
        }

        // The pending map should have been drained so a retry can rebind
        // the request_id without colliding.
        assert!(super::super::super::lock_recovered(&pending).is_empty());
    }

    /// Untagged `op=error` envelopes (no `request_id`) still flow through
    /// the control channel, preserving the sequential-op contract used by
    /// health / capabilities / ensure_task.
    #[tokio::test]
    async fn untagged_error_envelope_flows_to_control_channel() {
        let pid = WorkerPid(12345);
        let pending = Arc::new(std::sync::Mutex::new(HashMap::<
            String,
            oneshot::Sender<ExecuteResponseV2>,
        >::new()));
        let control = Arc::new(tokio::sync::Mutex::new(
            None::<oneshot::Sender<WorkerControlResponse>>,
        ));

        let (ctrl_tx, ctrl_rx) = oneshot::channel();
        {
            let mut slot = control.lock().await;
            *slot = Some(ctrl_tx);
        }

        let envelope = "{\"op\":\"error\",\"error\":\"capabilities-time crash\"}\n";
        let mut reader = BufReader::new(envelope.as_bytes());

        reader_loop_generic(&mut reader, pending.clone(), control.clone(), pid).await;

        let routed = ctrl_rx
            .await
            .expect("control channel must receive untagged error");
        match routed {
            WorkerControlResponse::Error(msg) => {
                assert!(msg.contains("capabilities-time crash"));
            }
            other => panic!("expected WorkerControlResponse::Error, got {other:?}"),
        }
    }
}
