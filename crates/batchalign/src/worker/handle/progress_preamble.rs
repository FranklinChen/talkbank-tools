//! Read helper: consume `progress_v2` preamble events while waiting for
//! the final IPC response.
//!
//! The Python worker emits `{"op": "progress_v2", ...}` lines on stdout
//! during long-running cache-relevant operations: Stanza catalog
//! download, Stanza per-language pack download, HuggingFace model
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

use tokio::io::{AsyncBufRead, AsyncBufReadExt};
use tokio::sync::mpsc;

use crate::api::PositiveSeconds;
use crate::types::worker_v2::ProgressEventV2;
use crate::worker::error::{WorkerError, WorkerWait};

use super::WorkerHandle;
use super::protocol::WorkerResponse;
use serde::Deserialize;

impl WorkerHandle {
    /// Read JSON-line responses from `reader` until either the deadline
    /// elapses or a non-`ProgressV2` `WorkerResponse` arrives.
    ///
    /// Each `ProgressV2` event encountered is optionally forwarded to
    /// `progress_tx` (drops silently if the channel is closed or full)
    /// and the loop continues. The first non-progress response is
    /// returned to the caller, which then matches on the variants it
    /// expects (`Capabilities`, `Health`, `Infer`, etc.) the same way
    /// it would have if it had called `read_response()` directly.
    ///
    /// This is the static, reader-driven testable shape. The
    /// instance-method twin
    /// [`Self::read_response_skipping_progress_via_self`] keeps the
    /// stderr-drain / `ProcessExited` attribution that
    /// [`Self::read_response`] performs on pipe errors; production
    /// sites use that one. The static shape exists so the loop
    /// discipline is unit-testable without spawning a child process.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(super) async fn read_response_skipping_progress<R>(
        reader: &mut R,
        waited_for: WorkerWait,
        limit: PositiveSeconds,
        progress_tx: Option<&mpsc::Sender<ProgressEventV2>>,
    ) -> Result<WorkerResponse, WorkerError>
    where
        R: AsyncBufRead + Unpin,
    {
        let deadline = crate::worker::Deadline::after(limit);
        let mut noise = super::protocol::NoiseRun::default();
        loop {
            let mut line = String::new();
            let read_fut = reader.read_line(&mut line);
            // Wrap the per-line read in `timeout_at` so a stalled
            // worker can never hold the supervisor forever, regardless
            // of how many progress events it has already emitted.
            let bytes = match deadline.within(read_fut).await {
                Some(Ok(b)) => b,
                Some(Err(io_err)) => {
                    return Err(WorkerError::Protocol(format!(
                        "I/O error reading response: {io_err}"
                    )));
                }
                None => return Err(WorkerError::Timeout { waited_for, limit }),
            };
            if bytes == 0 {
                return Err(WorkerError::Protocol(
                    "worker closed stdout before delivering a final response".into(),
                ));
            }

            // The one line rule every reader applies.
            let message = match super::protocol::WireLine::classify(&line) {
                super::protocol::WireLine::Blank => continue,
                super::protocol::WireLine::Noise => {
                    noise
                        .noise(&line)
                        .map_err(super::protocol::NoiseLimit::into_worker_error)?;
                    continue;
                }
                super::protocol::WireLine::Message(message) => message,
            };
            noise.message();
            let response = WorkerResponse::deserialize(&message).map_err(|e| {
                WorkerError::Protocol(format!(
                    "failed to decode worker response: {e} (line: {})",
                    super::protocol::excerpt(line.trim())
                ))
            })?;

            match response {
                WorkerResponse::ProgressV2 { event } => {
                    // Forward to the optional progress channel. Drop
                    // silently if the channel is closed or full
                    // progress is observability, not flow control, and
                    // a slow consumer must not stall the worker.
                    if let Some(tx) = progress_tx {
                        let _ = tx.try_send(event);
                    }
                    continue;
                }
                other => return Ok(other),
            }
        }
    }

    /// Instance-method twin of [`Self::read_response_skipping_progress`].
    ///
    /// Production sites use this one because it delegates to
    /// [`Self::read_response`], which preserves the BrokenPipe →
    /// `drain_stderr_tail` → `WorkerError::ProcessExited` attribution
    /// the static helper cannot do (it has no access to the child or
    /// stderr handle). The dispatch contract, skip every
    /// `ProgressV2`, optionally forward to the channel, return the
    /// first non-progress response: is identical.
    pub(super) async fn read_response_skipping_progress_via_self(
        &mut self,
        waited_for: WorkerWait,
        limit: PositiveSeconds,
        progress_tx: Option<&mpsc::Sender<ProgressEventV2>>,
    ) -> Result<WorkerResponse, WorkerError> {
        let deadline = crate::worker::Deadline::after(limit);
        loop {
            let response = match deadline.within(self.read_response()).await {
                Some(read) => read?,
                None => return Err(WorkerError::Timeout { waited_for, limit }),
            };

            match response {
                WorkerResponse::ProgressV2 { event } => {
                    if let Some(tx) = progress_tx {
                        let _ = tx.try_send(event);
                    }
                    continue;
                }
                other => {
                    // A complete terminal response was read: safe to reuse.
                    self.request_flight = super::RequestFlight::Idle;
                    return Ok(other);
                }
            }
        }
    }
}

#[cfg(test)]
mod progress_preamble_tests {
    use super::*;
    use crate::types::worker_v2::WorkerRequestIdV2;
    use tokio::io::BufReader;

    /// Wall-clock window large enough that no test should ever miss it.
    /// A limit large enough that no test should ever reach it.
    const GENEROUS: PositiveSeconds = PositiveSeconds::literal::<10>();

    /// A few canned `progress_v2` lines covering the cache classes the
    /// Python worker actually emits at boundary events: Stanza catalog
    /// (capabilities-probe time), Stanza per-language pack
    /// (ensure_task time), and HuggingFace download (Whisper / pyannote
    /// / transformers warm-up time).
    const STANZA_CATALOG_PROGRESS: &str = concat!(
        r#"{"op": "progress_v2", "event": {"request_id": "", "completed": 0, "total": 0, "stage": "downloading_stanza_catalog"}}"#,
        "\n",
    );
    const STANZA_LANG_PROGRESS: &str = concat!(
        r#"{"op": "progress_v2", "event": {"request_id": "", "completed": 0, "total": 0, "stage": "downloading_stanza_lang_zh"}}"#,
        "\n",
    );
    const HF_MODEL_PROGRESS: &str = concat!(
        r#"{"op": "progress_v2", "event": {"request_id": "", "completed": 0, "total": 0, "stage": "downloading_hf_openai_whisper-large-v3"}}"#,
        "\n",
    );

    /// Concrete final-response JSON fragments matching what the Python
    /// worker emits for each request type. We deliberately pick minimal
    /// payloads: the shape only has to deserialize, not be
    /// semantically meaningful.
    const HEALTH_RESPONSE: &str = concat!(
        r#"{"op": "health", "response": {"status": "ok", "command": "infer:morphosyntax", "lang": "eng", "pid": 4242, "uptime_s": 12.5}}"#,
        "\n",
    );
    const CAPABILITIES_RESPONSE: &str = concat!(
        r#"{"op": "capabilities", "response": {"commands": ["morphotag"], "free_threaded": false, "infer_tasks": ["morphosyntax"], "engine_versions": {"morphosyntax": null}}}"#,
        "\n",
    );

    #[tokio::test(start_paused = true)]
    async fn runtime_regression_capabilities_accept_cold_imports_and_remain_bounded() {
        use tokio::io::AsyncWriteExt;
        for delay in [120, 301] {
            let (reader, mut writer) = tokio::io::duplex(4096);
            let sender = tokio::spawn(async move {
                tokio::time::sleep(std::time::Duration::from_secs(delay)).await;
                let _ = writer.write_all(CAPABILITIES_RESPONSE.as_bytes()).await;
            });
            let mut reader = BufReader::new(reader);
            let result = WorkerHandle::read_response_skipping_progress(
                &mut reader,
                WorkerWait::Capabilities,
                crate::worker::CAPABILITY_TIMEOUT,
                None,
            )
            .await;
            if delay == 120 {
                assert!(matches!(result, Ok(WorkerResponse::Capabilities { .. })));
            } else {
                assert!(result.is_err());
            }
            sender.abort();
            let _ = sender.await;
        }
    }
    const ENSURE_TASK_RESPONSE: &str = concat!(
        r#"{"op": "ensure_task", "response": {"status": "loaded", "task": "morphosyntax", "elapsed_s": 0.0}}"#,
        "\n",
    );
    const INFER_RESPONSE: &str = concat!(
        r#"{"op": "infer", "response": {"outcome": {"kind": "produced", "result": {}}, "elapsed_s": 0.5}}"#,
        "\n",
    );
    const BATCH_INFER_RESPONSE: &str = concat!(
        r#"{"op": "batch_infer", "response": {"results": []}}"#,
        "\n",
    );
    const ERROR_RESPONSE: &str =
        "{\"op\": \"error\", \"error\": \"boom\", \"kind\": \"runtime\"}\n";

    // ---- One test per cache-relevant request type --------------------
    // Each asserts: progress preamble of varying length, then the final
    // response, round-trips through the helper and lands as the
    // expected `WorkerResponse` variant.

    #[tokio::test]
    async fn returns_capabilities_after_catalog_progress() {
        let stdout = format!("{STANZA_CATALOG_PROGRESS}{CAPABILITIES_RESPONSE}");
        let mut r = BufReader::new(stdout.as_bytes());
        let resp = WorkerHandle::read_response_skipping_progress(
            &mut r,
            WorkerWait::Health,
            GENEROUS,
            None,
        )
        .await
        .expect("helper must consume progress preamble and return capabilities");
        assert!(matches!(resp, WorkerResponse::Capabilities { .. }));
    }

    #[tokio::test]
    async fn returns_ensure_task_after_lang_pack_progress() {
        let stdout = format!("{STANZA_LANG_PROGRESS}{ENSURE_TASK_RESPONSE}");
        let mut r = BufReader::new(stdout.as_bytes());
        let resp = WorkerHandle::read_response_skipping_progress(
            &mut r,
            WorkerWait::Health,
            GENEROUS,
            None,
        )
        .await
        .expect("helper must consume Stanza per-language download preamble");
        assert!(matches!(resp, WorkerResponse::EnsureTask { .. }));
    }

    #[tokio::test]
    async fn returns_health_after_progress() {
        let stdout = format!("{STANZA_CATALOG_PROGRESS}{HEALTH_RESPONSE}");
        let mut r = BufReader::new(stdout.as_bytes());
        let resp = WorkerHandle::read_response_skipping_progress(
            &mut r,
            WorkerWait::Health,
            GENEROUS,
            None,
        )
        .await
        .expect("health probes must tolerate progress preamble too");
        assert!(matches!(resp, WorkerResponse::Health { .. }));
    }

    #[tokio::test]
    async fn returns_infer_after_progress() {
        let stdout = format!("{HF_MODEL_PROGRESS}{INFER_RESPONSE}");
        let mut r = BufReader::new(stdout.as_bytes());
        let resp = WorkerHandle::read_response_skipping_progress(
            &mut r,
            WorkerWait::Health,
            GENEROUS,
            None,
        )
        .await
        .expect("legacy infer must tolerate HF-model warm-up preamble");
        assert!(matches!(resp, WorkerResponse::Infer { .. }));
    }

    #[tokio::test]
    async fn returns_batch_infer_after_progress() {
        let stdout = format!("{HF_MODEL_PROGRESS}{BATCH_INFER_RESPONSE}");
        let mut r = BufReader::new(stdout.as_bytes());
        let resp = WorkerHandle::read_response_skipping_progress(
            &mut r,
            WorkerWait::Health,
            GENEROUS,
            None,
        )
        .await
        .expect("legacy batch_infer must tolerate HF-model warm-up preamble");
        assert!(matches!(resp, WorkerResponse::BatchInfer { .. }));
    }

    #[tokio::test]
    async fn returns_first_response_when_no_preamble_present() {
        let mut r = BufReader::new(CAPABILITIES_RESPONSE.as_bytes());
        let resp = WorkerHandle::read_response_skipping_progress(
            &mut r,
            WorkerWait::Health,
            GENEROUS,
            None,
        )
        .await
        .expect("zero-progress preamble is the warm-cache happy path");
        assert!(matches!(resp, WorkerResponse::Capabilities { .. }));
    }

    // ---- Stacked preambles cover the multi-stage Stanza+HF case ------

    #[tokio::test]
    async fn returns_response_after_stacked_preamble_of_three() {
        let stdout = format!(
            "{STANZA_CATALOG_PROGRESS}{STANZA_LANG_PROGRESS}{HF_MODEL_PROGRESS}{ENSURE_TASK_RESPONSE}"
        );
        let mut r = BufReader::new(stdout.as_bytes());
        let resp = WorkerHandle::read_response_skipping_progress(
            &mut r,
            WorkerWait::Health,
            GENEROUS,
            None,
        )
        .await
        .expect("multi-stage preamble must drain cleanly");
        assert!(matches!(resp, WorkerResponse::EnsureTask { .. }));
    }

    // ---- Channel forwarding and absence-of-channel ------------------

    #[tokio::test]
    async fn forwards_progress_events_to_channel_when_provided() {
        let (tx, mut rx) = mpsc::channel::<ProgressEventV2>(8);
        let stdout =
            format!("{STANZA_CATALOG_PROGRESS}{STANZA_LANG_PROGRESS}{CAPABILITIES_RESPONSE}");
        let mut r = BufReader::new(stdout.as_bytes());
        let resp = WorkerHandle::read_response_skipping_progress(
            &mut r,
            WorkerWait::Health,
            GENEROUS,
            Some(&tx),
        )
        .await
        .expect("forwarding case must still return the final response");
        assert!(matches!(resp, WorkerResponse::Capabilities { .. }));

        // Drain everything the helper queued. Both progress events
        // should be present in order.
        drop(tx);
        let mut stages = Vec::new();
        while let Some(event) = rx.recv().await {
            stages.push(event.stage);
        }
        assert_eq!(
            stages.iter().map(String::as_str).collect::<Vec<_>>(),
            vec!["downloading_stanza_catalog", "downloading_stanza_lang_zh"],
            "all progress events must reach the channel in arrival order"
        );
    }

    #[tokio::test]
    async fn drops_progress_events_silently_when_no_channel_provided() {
        let stdout = format!("{STANZA_CATALOG_PROGRESS}{STANZA_LANG_PROGRESS}{HEALTH_RESPONSE}");
        let mut r = BufReader::new(stdout.as_bytes());
        // No tx provided; helper must still complete cleanly.
        let resp = WorkerHandle::read_response_skipping_progress(
            &mut r,
            WorkerWait::Health,
            GENEROUS,
            None,
        )
        .await
        .expect("missing progress channel must not panic or error");
        assert!(matches!(resp, WorkerResponse::Health { .. }));
    }

    // ---- Error / EOF / timeout edge cases ----------------------------

    #[tokio::test]
    async fn surfaces_worker_error_response_directly() {
        // Error responses must reach the caller as-is; they are not
        // progress to skip. The caller decides how to interpret kind.
        let mut r = BufReader::new(ERROR_RESPONSE.as_bytes());
        let resp = WorkerHandle::read_response_skipping_progress(
            &mut r,
            WorkerWait::Health,
            GENEROUS,
            None,
        )
        .await
        .expect("Error response is a final response");
        match resp {
            WorkerResponse::Error(failure) => {
                assert_eq!(failure.message, "boom");
            }
            other => panic!("expected Error variant, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn returns_protocol_error_on_eof_before_final_response() {
        // Unbounded preamble that closes without ever delivering a
        // final response. Helper must surface a Protocol error rather
        // than hanging or returning silently.
        let stdout = format!("{STANZA_CATALOG_PROGRESS}{STANZA_LANG_PROGRESS}");
        let mut r = BufReader::new(stdout.as_bytes());
        let result = WorkerHandle::read_response_skipping_progress(
            &mut r,
            WorkerWait::Health,
            GENEROUS,
            None,
        )
        .await;
        assert!(
            matches!(result, Err(WorkerError::Protocol(_))),
            "EOF mid-preamble must error, got {result:?}"
        );
    }

    /// A reader that never delivers a line ends at the limit with a typed
    /// timeout naming what was awaited, which the retry policy reads without
    /// parsing a message.
    #[tokio::test(start_paused = true)]
    async fn returns_a_typed_timeout_when_the_limit_passes() {
        // The writer half is held open and never written, so the read never
        // completes; paused time advances to the limit at once.
        let (reader, _writer) = tokio::io::duplex(64);
        let mut reader = BufReader::new(reader);
        let result = WorkerHandle::read_response_skipping_progress(
            &mut reader,
            WorkerWait::EnsureTask {
                task: crate::worker::InferTask::Morphosyntax,
            },
            PositiveSeconds::literal::<30>(),
            None,
        )
        .await;
        match result {
            Err(WorkerError::Timeout { waited_for, limit }) => {
                assert_eq!(
                    waited_for,
                    WorkerWait::EnsureTask {
                        task: crate::worker::InferTask::Morphosyntax
                    }
                );
                assert_eq!(limit.get(), 30);
            }
            other => panic!("an elapsed limit must be a typed timeout, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn rejects_undecodable_json_with_protocol_error() {
        // A JSON object that doesn't match any WorkerResponse variant
        // must surface as a Protocol error, silently dropping it
        // would mask wire-protocol bugs.
        let stdout = "{\"op\": \"this-is-not-a-real-variant\", \"foo\": 1}\n";
        let mut r = BufReader::new(stdout.as_bytes());
        let result = WorkerHandle::read_response_skipping_progress(
            &mut r,
            WorkerWait::Health,
            GENEROUS,
            None,
        )
        .await;
        assert!(
            matches!(result, Err(WorkerError::Protocol(_))),
            "unknown variant must error, got {result:?}"
        );
    }

    /// The sequential reader applies the shared line rule: a run of noise
    /// lines (text, a JSON scalar) up to the limit is skipped, and one more
    /// makes the stream untrusted, a retryable `OutputNoise`.
    #[tokio::test]
    async fn a_run_of_noise_lines_reaching_the_limit_is_retryable_output_noise() {
        use crate::runner::util::{classify_worker_error, is_retryable_worker_failure};
        let below = format!(
            "{}{}",
            "banner\n42\n"
                .repeat((super::super::protocol::MAX_RESPONSE_STDOUT_NOISE_LINES - 1) / 2),
            "{\"op\": \"shutdown\"}\n"
        );
        let mut r = BufReader::new(below.as_bytes());
        let ok = WorkerHandle::read_response_skipping_progress(
            &mut r,
            WorkerWait::Health,
            GENEROUS,
            None,
        )
        .await;
        assert!(matches!(ok, Ok(WorkerResponse::Shutdown)), "{ok:?}");

        let at_limit = "banner\n".repeat(super::super::protocol::MAX_RESPONSE_STDOUT_NOISE_LINES);
        let mut r = BufReader::new(at_limit.as_bytes());
        let error = WorkerHandle::read_response_skipping_progress(
            &mut r,
            WorkerWait::Health,
            GENEROUS,
            None,
        )
        .await
        .expect_err("the limit closes the stream");
        assert!(
            matches!(error, WorkerError::OutputNoise { .. }),
            "{error:?}"
        );
        assert!(is_retryable_worker_failure(classify_worker_error(&error)));
    }

    /// A limit too long for the clock is a wait with no end, not a panic.
    #[tokio::test]
    async fn a_limit_past_the_clock_does_not_overflow_the_deadline() {
        let limit = PositiveSeconds::new(std::num::NonZeroU64::MAX);
        let deadline = crate::worker::Deadline::after(limit);
        assert_eq!(deadline.within(async { 7 }).await, Some(7));
    }

    // Suppress unused-import warnings until GREEN.
    #[allow(dead_code)]
    fn _unused(_: WorkerRequestIdV2) {}
}
