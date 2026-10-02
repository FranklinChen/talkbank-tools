//! The response reader shared by the stdio and TCP shared GPU transports.
//!
//! One task per worker reads JSON lines and routes each to its owner:
//!
//! | Line | Owner | Owner receives |
//! |------|-------|----------------|
//! | `execute_v2` that decodes | the dispatch its `request_id` names | the response |
//! | `execute_v2` that does not decode | the dispatch its raw `request_id` names | `WorkerError::Protocol` |
//! | `op=error` with `request_id` | the sequential op whose request carried that id, else that dispatch | the reported failure's error (`ReportedFailure::into_worker_error`), or `Protocol` if the envelope does not decode |
//! | `op=error` without `request_id` | the waiting sequential op | the failure, through the control slot |
//! | `capabilities` / `ensure_task` / `shutdown` | the waiting sequential op of that op (for `ensure_task`, that task) | the reply, or a refusal if it does not decode |
//! | `health` (no shared GPU op asks for it) | the waiting sequential op | a refusal |
//! | an op the protocol does not name | the dispatch it names, else the sequential op | a refusal |
//!
//! A line whose owner is not waiting reaches nobody else: a failure is never
//! handed to a different dispatch, a late reply or failure of a sequential op
//! that timed out never answers the next op ([`ControlSlot`]), and a line
//! that names no owner fails no dispatch. Each dispatch is answered by its own
//! reply or by the stream closing.
//!
//! A line is classified by the rule every reader applies
//! ([`WireLine`]): a JSON object is a message, anything else is noise. The
//! stream closes at EOF, on a read error, after
//! `MAX_RESPONSE_STDOUT_NOISE_LINES` consecutive noise lines (retryably: the worker is retired), or when the
//! task ends any other way. Closing answers every waiter with the
//! reason ([`StreamClosed`]) and refuses later registrations, which is what
//! the worker's liveness check reads.

use serde::Deserialize;
use serde_json::Value;
use tokio::io::{AsyncBufRead, AsyncBufReadExt};
use tracing::{Instrument, debug, error, warn};

use crate::worker::handle::{NoiseRun, WireLine};

use crate::types::worker_v2::WorkerRequestIdV2;
use crate::worker::WorkerPid;
use crate::worker::handle::ReportedFailure;

use super::envelopes::{
    CapabilitiesResponseEnvelope, EnsureTaskResponseEnvelope, ExecuteResponseV2Envelope,
};
use super::routes::{
    ControlLine, ControlOpName, ControlReply, ControlSlot, PendingDispatches, ReplyFailure,
    StreamClosed,
};

/// The two places a worker's lines are routed to.
#[derive(Clone, Default)]
pub(crate) struct Routes {
    /// V2 dispatches, by request id.
    pub(crate) pending: PendingDispatches,
    /// The one sequential op.
    pub(crate) control: ControlSlot,
}

impl Routes {
    /// Close both ends with `reason`.
    fn close(&self, reason: StreamClosed) {
        self.pending.close(reason.clone());
        self.control.close(reason);
    }
}

/// Closes the routes when the reader task ends, however it ends: with the
/// reader's own reason when it returns one, with
/// [`StreamClosed::ReaderStopped`] when the task is aborted or panics.
struct CloseOnExit(Option<Routes>);

impl CloseOnExit {
    fn finish(mut self, reason: StreamClosed) {
        if let Some(routes) = self.0.take() {
            routes.close(reason);
        }
    }
}

impl Drop for CloseOnExit {
    fn drop(&mut self) {
        if let Some(routes) = self.0.take() {
            routes.close(StreamClosed::ReaderStopped);
        }
    }
}

/// Spawn the reader task for one worker's output.
pub(crate) fn spawn_reader<R>(
    mut reader: R,
    routes: Routes,
    pid: WorkerPid,
) -> tokio::task::JoinHandle<()>
where
    R: AsyncBufRead + Unpin + Send + 'static,
{
    // A named span so tokio-console attributes the loop to its worker, and
    // its log lines carry the pid.
    let span = tracing::info_span!("shared_gpu_reader_loop", pid = %pid);
    // Built outside the task and moved in, so even a task aborted before its
    // first poll drops the guard and closes the routes.
    let guard = CloseOnExit(Some(routes.clone()));
    tokio::spawn(
        async move {
            let reason = read_until_closed(&mut reader, &routes, pid).await;
            guard.finish(reason);
        }
        .instrument(span),
    )
}

/// Read and route lines until the stream closes; return why it closed.
async fn read_until_closed<R: AsyncBufRead + Unpin>(
    reader: &mut R,
    routes: &Routes,
    pid: WorkerPid,
) -> StreamClosed {
    let mut line = String::new();
    let mut noise = NoiseRun::default();
    loop {
        line.clear();
        match reader.read_line(&mut line).await {
            Ok(0) => {
                debug!(pid = %pid, "GPU worker stream closed (EOF)");
                return StreamClosed::Eof;
            }
            Err(error) => {
                error!(pid = %pid, %error, "GPU worker: stream read error");
                return StreamClosed::ReadFailed(error.to_string());
            }
            Ok(_) => {}
        }
        // The one line rule every reader applies.
        match WireLine::classify(&line) {
            WireLine::Blank => {}
            WireLine::Message(message) => {
                noise.message();
                route_line(&message, routes, pid);
            }
            WireLine::Noise => {
                if let Err(limit) = noise.noise(&line) {
                    return StreamClosed::Noise(limit);
                }
            }
        }
    }
}

/// The request id a line names at `pointer`, read from the raw JSON so a line
/// whose envelope the protocol refuses can still reach its owner.
fn named_request(parsed: &Value, pointer: &str) -> Option<WorkerRequestIdV2> {
    parsed
        .pointer(pointer)
        .and_then(Value::as_str)
        .map(WorkerRequestIdV2::from)
}

/// Route one parsed line to its owner.
fn route_line(parsed: &Value, routes: &Routes, pid: WorkerPid) {
    let op = parsed.get("op").and_then(Value::as_str);
    match op {
        Some("execute_v2") => match ExecuteResponseV2Envelope::deserialize(parsed) {
            Ok(envelope) => {
                let request_id = envelope.response.request_id().clone();
                if routes
                    .pending
                    .answer(&request_id, Ok(envelope.response))
                    .is_err()
                {
                    warn!(
                        pid = %pid,
                        request_id = %request_id,
                        "GPU worker: execute_v2 response has no pending dispatch"
                    );
                }
            }
            Err(error) => {
                // A refused response still resolves its dispatch at once,
                // rather than leaving it on its per-request timeout.
                let failure =
                    ReplyFailure::Refused(format!("failed to decode execute_v2 response: {error}"));
                match named_request(parsed, "/response/request_id") {
                    Some(request_id) => fail_dispatch(routes, pid, &request_id, failure),
                    // An execute_v2 line is never a sequential op's reply, and
                    // its owner is unknown: no dispatch is failed on it.
                    None => error!(
                        pid = %pid,
                        error = %failure.message(),
                        "GPU worker: refused execute_v2 line names no request; \
                         no dispatch is failed on it"
                    ),
                }
            }
        },
        Some("error") => {
            let failure = match ReportedFailure::deserialize(parsed) {
                Ok(reported) => ReplyFailure::Reported(reported),
                Err(error) => {
                    ReplyFailure::Refused(format!("failed to decode error response: {error}"))
                }
            };
            match named_request(parsed, "/request_id") {
                Some(request_id) => fail_owner(routes, pid, &request_id, failure),
                None => offer_control(routes, pid, ControlLine::Unowned(failure)),
            }
        }
        // No shared GPU op asks for health: the pool watches a shared worker
        // through its stream's liveness, not a probe.
        Some("health") => offer_control(
            routes,
            pid,
            ControlLine::Unowned(ReplyFailure::Refused(
                "health reply, but no shared GPU op asks for health".into(),
            )),
        ),
        Some("capabilities") => offer_control(
            routes,
            pid,
            decode_control(
                parsed,
                ControlOpName::Capabilities,
                |envelope: CapabilitiesResponseEnvelope| {
                    ControlReply::Capabilities(envelope.response)
                },
            ),
        ),
        Some("ensure_task") => offer_control(
            routes,
            pid,
            decode_control(
                parsed,
                ControlOpName::EnsureTask,
                |envelope: EnsureTaskResponseEnvelope| ControlReply::EnsureTask(envelope.response),
            ),
        ),
        Some("shutdown") => offer_control(routes, pid, ControlLine::Reply(ControlReply::Shutdown)),
        unknown => {
            let failure = ReplyFailure::Refused(format!(
                "GPU worker sent an op this protocol does not name: {unknown:?}"
            ));
            let owner = named_request(parsed, "/request_id")
                .or_else(|| named_request(parsed, "/response/request_id"));
            match owner {
                Some(request_id) => fail_owner(routes, pid, &request_id, failure),
                None => offer_control(routes, pid, ControlLine::Unowned(failure)),
            }
        }
    }
}

/// A sequential op's reply, or the refusal of one that does not decode.
fn decode_control<E: serde::de::DeserializeOwned>(
    parsed: &Value,
    op: ControlOpName,
    reply: impl FnOnce(E) -> ControlReply,
) -> ControlLine {
    match E::deserialize(parsed) {
        Ok(envelope) => ControlLine::Reply(reply(envelope)),
        Err(error) => ControlLine::Undecodable {
            op,
            detail: format!("failed to decode {} response: {error}", op.wire_name()),
        },
    }
}

/// Fail the owner a tagged failure names: the waiting control op when its
/// request carried that id, else the dispatch of that id.
fn fail_owner(routes: &Routes, pid: WorkerPid, request_id: &str, failure: ReplyFailure) {
    match routes.control.deliver_tagged_failure(request_id, failure) {
        Ok(()) => debug!(
            pid = %pid,
            request_id,
            "GPU worker: failure routed to its control op"
        ),
        Err(failure) => fail_dispatch(routes, pid, request_id, failure),
    }
}

/// Fail the one dispatch a line names. Nobody waiting means that caller
/// already gave up or was answered; the failure goes nowhere else.
fn fail_dispatch(routes: &Routes, pid: WorkerPid, request_id: &str, failure: ReplyFailure) {
    let message = failure.message();
    let error = failure.into_worker_error(ReportedFailure::into_worker_error);
    match routes.pending.answer(request_id, Err(error)) {
        Ok(()) => debug!(
            pid = %pid,
            request_id,
            error = %message,
            "GPU worker: failure routed to its dispatch"
        ),
        Err(_) => warn!(
            pid = %pid,
            request_id,
            error = %message,
            "GPU worker: failure names a request nobody is waiting for; dropped"
        ),
    }
}

/// Offer a line to the waiting sequential op. A line no waiting op claims
/// (nobody waits, or it answers an op that already timed out) is logged and
/// fails nobody.
fn offer_control(routes: &Routes, pid: WorkerPid, line: ControlLine) {
    if let Err(unclaimed) = routes.control.deliver(line) {
        match unclaimed {
            ControlLine::Reply(reply) => warn!(
                pid = %pid,
                reply = ?reply,
                "GPU worker: sequential reply no waiting op asked for (late, or another op's); dropped"
            ),
            ControlLine::Undecodable { op, detail } => error!(
                pid = %pid,
                op = op.wire_name(),
                error = %detail,
                "GPU worker: undecodable sequential reply no waiting op asked for; dropped"
            ),
            ControlLine::Unowned(failure) => error!(
                pid = %pid,
                error = %failure.message(),
                "GPU worker: failure names no request and no sequential op is waiting; \
                 no dispatch is failed on it"
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runner::util::{classify_worker_error, is_retryable_worker_failure};
    use crate::types::worker_v2::WorkerErrorKind;
    use crate::worker::InferTask;
    use crate::worker::error::WorkerError;
    use crate::worker::handle::MAX_RESPONSE_STDOUT_NOISE_LINES;
    use std::time::Duration;
    use tokio::io::BufReader;

    const PID: WorkerPid = WorkerPid(12345);

    /// Register one waiting dispatch under `request_id`.
    fn wait_for(routes: &Routes, request_id: &str) -> super::super::routes::DispatchReceipt {
        routes
            .pending
            .register(WorkerRequestIdV2::from(request_id))
            .expect("an open stream registers")
    }

    /// Route one line without closing the stream.
    fn route(routes: &Routes, line: &str) {
        let parsed: Value = serde_json::from_str(line).expect("test lines are JSON");
        route_line(&parsed, routes, PID);
    }

    /// Feed `lines` through the reader to EOF and close the routes with the
    /// reason, as the spawned task does.
    async fn read_to_end(routes: &Routes, lines: &str) -> StreamClosed {
        let mut reader = BufReader::new(lines.as_bytes());
        let reason = read_until_closed(&mut reader, routes, PID).await;
        routes.close(reason.clone());
        reason
    }

    /// The reply a receipt holds once routing has answered it.
    async fn answered(receipt: super::super::routes::DispatchReceipt) -> DispatchReplyForTest {
        receipt
            .reply_within(Duration::from_secs(5))
            .await
            .expect("the dispatch must be answered, not left on its timeout")
    }
    type DispatchReplyForTest = super::super::routes::DispatchReply;

    /// An `execute_v2` line the strict response parse refuses still resolves
    /// its dispatch at once, with `WorkerError::Protocol`, as the sequential
    /// path's undecodable reply does.
    #[tokio::test]
    async fn refused_execute_v2_response_fails_its_dispatch_with_protocol_error() {
        let routes = Routes::default();
        let receipt = wait_for(&routes, "asr-v2-request-88");
        route(
            &routes,
            r#"{"op":"execute_v2","response":{"request_id":"asr-v2-request-88","outcome":{"kind":"success"},"elapsed_s":0.5}}"#,
        );
        match answered(receipt).await {
            Err(WorkerError::Protocol(message)) => assert!(
                message.contains("carried no result payload"),
                "the refusal detail must reach the caller, got: {message}"
            ),
            other => panic!("a refused response must be a protocol error, got {other:?}"),
        }
    }

    /// A refused `execute_v2` line that names no request fails no dispatch:
    /// its siblings are left to their own replies.
    #[tokio::test]
    async fn an_uncorrelated_refused_execute_v2_line_fails_no_sibling() {
        let routes = Routes::default();
        let first = wait_for(&routes, "fa-v2-request-1");
        let _second = wait_for(&routes, "fa-v2-request-2");
        route(
            &routes,
            r#"{"op":"execute_v2","response":{"outcome":{"kind":"success"}}}"#,
        );
        assert_eq!(routes.pending.waiting(), 2, "no sibling may be failed");
        assert!(
            first
                .reply_within(Duration::from_millis(20))
                .await
                .is_none(),
            "the sibling must still be waiting for its own reply"
        );
    }

    /// A tagged `invalid_request` refusal fails its dispatch with the terminal
    /// `WorkerError::RequestRefused`.
    #[tokio::test]
    async fn tagged_request_refusal_fails_its_dispatch_terminally() {
        let routes = Routes::default();
        let receipt = wait_for(&routes, "asr-v2-request-77");
        route(
            &routes,
            r#"{"op":"error","error":"invalid execute_v2 request: ValidationError","kind":"invalid_request","request_id":"asr-v2-request-77"}"#,
        );
        let error = match answered(receipt).await {
            Err(error) => error,
            Ok(response) => panic!("a worker error must not resolve as a response: {response:?}"),
        };
        assert!(
            matches!(&error, WorkerError::RequestRefused(message) if message.contains("ValidationError")),
            "{error:?}"
        );
        assert!(!is_retryable_worker_failure(classify_worker_error(&error)));
        assert_eq!(routes.pending.waiting(), 0);
    }

    /// An `op=error` line without `kind` is refused: its dispatch gets a
    /// protocol error, never a retryable runtime failure by default.
    #[tokio::test]
    async fn an_error_line_without_kind_is_a_protocol_refusal() {
        let routes = Routes::default();
        let receipt = wait_for(&routes, "asr-v2-request-6");
        route(
            &routes,
            r#"{"op":"error","error":"no kind","request_id":"asr-v2-request-6"}"#,
        );
        match answered(receipt).await {
            Err(WorkerError::Protocol(message)) => assert!(
                message.contains("kind"),
                "the refusal must name the missing field, got: {message}"
            ),
            other => panic!("expected a protocol refusal, got {other:?}"),
        }
    }

    /// A tagged `bootstrap` failure is terminal on this path as on the
    /// sequential one.
    #[tokio::test]
    async fn tagged_bootstrap_error_is_terminal() {
        let routes = Routes::default();
        let receipt = wait_for(&routes, "asr-v2-request-5");
        route(
            &routes,
            r#"{"op":"error","error":"model missing","kind":"bootstrap","request_id":"asr-v2-request-5"}"#,
        );
        let error = answered(receipt).await.expect_err("a reported failure");
        assert!(matches!(error, WorkerError::Bootstrap(_)), "got {error:?}");
        assert!(!is_retryable_worker_failure(classify_worker_error(&error)));
    }

    /// An untagged `op=error` that no sequential op is waiting for fails no
    /// dispatch: its owner is unknown, and the dispatches in flight are left
    /// to their own replies.
    #[tokio::test]
    async fn an_untagged_error_with_no_sequential_op_fails_no_dispatch() {
        let routes = Routes::default();
        let receipt = wait_for(&routes, "speaker-embedding-v2-request-1");
        route(
            &routes,
            r#"{"op":"error","error":"module has no attribute","kind":"runtime"}"#,
        );
        assert_eq!(routes.pending.waiting(), 1);
        assert!(
            receipt
                .reply_within(Duration::from_millis(20))
                .await
                .is_none()
        );
    }

    /// A TAGGED error for a request nobody is waiting for reaches nobody: not
    /// another dispatch, and not the sequential op waiting in the control
    /// slot (a health check would otherwise fail on a late dispatch error).
    #[tokio::test]
    async fn a_tagged_error_for_an_unknown_request_never_reaches_the_control_slot() {
        let routes = Routes::default();
        let receipt = wait_for(&routes, "asr-v2-request-99");
        let mut control = routes
            .control
            .arm_capabilities()
            .expect("an idle slot arms")
            .answer;
        route(
            &routes,
            r#"{"op":"error","error":"late failure","kind":"runtime","request_id":"asr-v2-request-98"}"#,
        );
        assert!(
            control.try_recv().is_err(),
            "a request-tagged error must not answer the sequential op"
        );
        assert!(
            receipt
                .reply_within(Duration::from_millis(20))
                .await
                .is_none()
        );
    }

    /// A late `ensure_task` reply for a task other than the one the waiting op
    /// asked for does not answer it: the op that sent it timed out, and the
    /// waiting op's own reply is still to come, and answers it.
    #[tokio::test]
    async fn a_late_ensure_task_reply_for_another_task_does_not_answer_the_waiting_op() {
        let routes = Routes::default();
        let mut control = routes
            .control
            .arm_ensure_task(InferTask::Morphosyntax)
            .expect("an idle slot arms")
            .answer;
        route(
            &routes,
            r#"{"op":"ensure_task","response":{"status":"loaded","task":"fa","elapsed_s":1.0}}"#,
        );
        assert!(
            control.try_recv().is_err(),
            "a reply naming fa must not answer an ensure_task(morphosyntax)"
        );
        route(
            &routes,
            r#"{"op":"ensure_task","response":{"status":"loaded","task":"morphosyntax","elapsed_s":1.0}}"#,
        );
        match control.await.expect("answered") {
            Ok(response) => assert_eq!(response.task, InferTask::Morphosyntax),
            Err(failure) => panic!("expected its own reply, got {failure:?}"),
        }
    }

    /// A late failure of an op that timed out does not answer the next op:
    /// the failure carries the id of the request that failed, and the
    /// waiting op's request carried another.
    #[tokio::test]
    async fn a_late_failure_of_a_timed_out_op_does_not_answer_the_next_op() {
        let routes = Routes::default();
        let timed_out = routes
            .control
            .arm_ensure_task(InferTask::Fa)
            .expect("an idle slot arms");
        let stale_id = timed_out.request_id.to_string();
        drop(timed_out.answer);
        let next = routes
            .control
            .arm_ensure_task(InferTask::Fa)
            .expect("a slot whose waiter left re-arms");
        let mut answer = next.answer;
        route(
            &routes,
            &format!(
                r#"{{"op":"error","error":"load failed","kind":"bootstrap","request_id":"{stale_id}"}}"#
            ),
        );
        assert!(
            answer.try_recv().is_err(),
            "a failure tagged with the timed-out op's id must not answer the next op"
        );
        route(
            &routes,
            &format!(
                r#"{{"op":"error","error":"load failed","kind":"bootstrap","request_id":"{}"}}"#,
                next.request_id
            ),
        );
        assert!(matches!(
            answer.await.expect("answered"),
            Err(ReplyFailure::Reported(_))
        ));
    }

    /// A capabilities reply does not answer an `ensure_task`, and the
    /// reverse: each waiter takes only its own op's reply.
    #[tokio::test]
    async fn a_reply_of_another_op_does_not_answer_the_waiting_op() {
        let routes = Routes::default();
        let mut control = routes
            .control
            .arm_ensure_task(InferTask::Fa)
            .expect("an idle slot arms")
            .answer;
        route(
            &routes,
            r#"{"op":"capabilities","response":{"commands":[],"free_threaded":false,"infer_tasks":["fa"],"engine_versions":{"fa":null}}}"#,
        );
        assert!(control.try_recv().is_err());
    }

    /// An untagged `op=error` goes to the waiting sequential op, with its kind.
    #[tokio::test]
    async fn an_untagged_error_answers_the_sequential_op_with_its_kind() {
        let routes = Routes::default();
        let control = routes
            .control
            .arm_capabilities()
            .expect("an idle slot arms")
            .answer;
        route(
            &routes,
            r#"{"op":"error","error":"capabilities-time crash","kind":"bootstrap"}"#,
        );
        match control.await.expect("answered") {
            Err(ReplyFailure::Reported(failure)) => {
                assert_eq!(failure.kind, WorkerErrorKind::Bootstrap);
                assert!(failure.message.contains("capabilities-time crash"));
            }
            other => panic!("expected the reported failure, got {other:?}"),
        }
    }

    /// At EOF every waiter is answered through its own reply with
    /// `ProcessExited`, the sequential op included, and the worker reports
    /// itself unavailable.
    #[tokio::test]
    async fn eof_answers_every_waiter_and_marks_the_worker_dead() {
        let routes = Routes::default();
        let first = wait_for(&routes, "fa-v2-request-1");
        let second = wait_for(&routes, "fa-v2-request-2");
        let control = routes
            .control
            .arm_capabilities()
            .expect("an idle slot arms")
            .answer;

        assert_eq!(read_to_end(&routes, "").await, StreamClosed::Eof);

        for receipt in [first, second] {
            assert!(matches!(
                answered(receipt).await,
                Err(WorkerError::ProcessExited { .. })
            ));
        }
        assert!(matches!(
            control.await.expect("answered"),
            Err(ReplyFailure::StreamClosed(StreamClosed::Eof))
        ));
        assert!(matches!(
            routes.pending.liveness(),
            Err(WorkerError::ProcessExited { .. })
        ));
    }

    /// The reader stops trusting a stream after the sequential path's noise
    /// limit of consecutive non-JSON lines.
    #[tokio::test]
    async fn too_many_non_json_lines_close_the_stream() {
        let routes = Routes::default();
        let receipt = wait_for(&routes, "fa-v2-request-1");
        let noise = "not json\n".repeat(MAX_RESPONSE_STDOUT_NOISE_LINES + 3);
        let reason = read_to_end(&routes, &noise).await;
        assert!(matches!(reason, StreamClosed::Noise(_)), "{reason:?}");
        let error = answered(receipt).await.expect_err("a closed stream");
        assert!(
            matches!(error, WorkerError::OutputNoise { .. }),
            "{error:?}"
        );
        assert!(
            is_retryable_worker_failure(classify_worker_error(&error)),
            "the worker is retired and the request runs again: {error:?}"
        );
    }

    /// A JSON scalar or array is not a protocol message: it is noise, by the
    /// rule the sequential readers apply too.
    #[tokio::test]
    async fn a_json_scalar_line_is_noise() {
        let routes = Routes::default();
        let lines = "42\n".repeat(MAX_RESPONSE_STDOUT_NOISE_LINES);
        assert!(matches!(
            read_to_end(&routes, &lines).await,
            StreamClosed::Noise(_)
        ));
    }

    /// Fewer noise lines than the limit, separated by protocol lines, do not.
    #[tokio::test]
    async fn noise_below_the_limit_is_tolerated() {
        let routes = Routes::default();
        let lines = format!(
            "{}{{\"op\":\"shutdown\"}}\n{}",
            "noise\n".repeat(MAX_RESPONSE_STDOUT_NOISE_LINES - 1),
            "noise\n".repeat(MAX_RESPONSE_STDOUT_NOISE_LINES - 1),
        );
        assert_eq!(read_to_end(&routes, &lines).await, StreamClosed::Eof);
    }

    /// An op the protocol does not name, and a sequential reply that does not
    /// decode, answer the waiting sequential op with a refusal instead of
    /// leaving it on its timeout.
    #[tokio::test]
    async fn unknown_ops_and_undecodable_replies_answer_the_sequential_op() {
        let routes = Routes::default();
        let control = routes.control.arm_capabilities().expect("arm").answer;
        route(&routes, r#"{"op":"surprise"}"#);
        assert!(matches!(
            control.await.expect("answered"),
            Err(ReplyFailure::Refused(message)) if message.contains("surprise")
        ));

        let control = routes.control.arm_capabilities().expect("arm again").answer;
        route(&routes, r#"{"op":"capabilities","response":{"status":7}}"#);
        assert!(matches!(
            control.await.expect("answered"),
            Err(ReplyFailure::Refused(message)) if message.contains("capabilities")
        ));
    }

    /// Aborting the reader task closes the routes rather than leaving waiters
    /// on their timeouts.
    #[tokio::test]
    async fn an_aborted_reader_closes_the_routes() {
        let routes = Routes::default();
        let receipt = wait_for(&routes, "fa-v2-request-1");
        let (_keep_open, reader) = tokio::io::duplex(64);
        let task = spawn_reader(BufReader::new(reader), routes.clone(), PID);
        task.abort();
        let _ = task.await;
        assert!(matches!(
            answered(receipt).await,
            Err(WorkerError::ProcessExited { .. })
        ));
    }
}
