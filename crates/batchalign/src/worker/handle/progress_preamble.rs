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

use tokio::sync::mpsc;

use crate::types::worker_v2::ProgressEventV2;
use crate::worker::error::WorkerError;
use crate::worker::{ResponseWait, RunningWait};

use super::WorkerHandle;
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
pub(crate) trait ResponseSource {
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
/// `wait` ends.
///
/// Each `ProgressV2` event is shown to the wait (a loading wait re-arms on an
/// advance, a budget on nothing; [`RunningWait::observe`]) and optionally
/// forwarded to `progress_tx`, dropped if the channel is closed or full:
/// progress is observability, not flow control, and a slow consumer must not
/// stall the worker. The first other response is returned for the caller to
/// match on the variants it expects.
pub(crate) async fn read_skipping_progress<S: ResponseSource>(
    source: &mut S,
    wait: ResponseWait,
    progress_tx: Option<&mpsc::Sender<ProgressEventV2>>,
) -> Result<WorkerResponse, WorkerError> {
    let loading = crate::worker::pool::checkout_wait::ObservedModelLoad::for_wait(&wait);
    let mut running = RunningWait::start(wait);
    loop {
        let response = match running.within(source.read_response()).await {
            Some(read) => read?,
            None => return Err(running.expired()),
        };
        match response {
            WorkerResponse::ProgressV2 { event } => {
                running.observe(&event);
                if let (Some(loading), ProgressEventV2::Load(load)) = (&loading, &event) {
                    loading.report(load);
                }
                if let Some(tx) = progress_tx {
                    let _ = tx.try_send(event);
                }
            }
            other => return Ok(other),
        }
    }
}

impl WorkerHandle {
    /// [`read_skipping_progress`] on this worker, marking the request flight
    /// idle once a terminal response was read (the worker is then safe to
    /// reuse).
    pub(super) async fn read_response_skipping_progress_via_self(
        &mut self,
        wait: ResponseWait,
        progress_tx: Option<&mpsc::Sender<ProgressEventV2>>,
    ) -> Result<WorkerResponse, WorkerError> {
        let response = read_skipping_progress(self, wait, progress_tx).await?;
        self.request_flight = super::RequestFlight::Idle;
        Ok(response)
    }

    /// [`read_skipping_progress`] over a bare line reader, so the loop can be
    /// tested without spawning a worker.
    #[cfg(test)]
    pub(super) async fn read_response_skipping_progress<R>(
        reader: &mut R,
        wait: ResponseWait,
        progress_tx: Option<&mpsc::Sender<ProgressEventV2>>,
    ) -> Result<WorkerResponse, WorkerError>
    where
        R: tokio::io::AsyncBufRead + Unpin,
    {
        read_skipping_progress(&mut LineSource(reader), wait, progress_tx).await
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
mod progress_preamble_tests {
    use super::*;
    use crate::api::PositiveSeconds;
    use crate::types::worker_v2::WorkerRequestIdV2;
    use crate::worker::error::WorkerWait;
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
        r#"{"op": "progress_v2", "event": {"scope": "load", "stage": "downloading_stanza_catalog", "measure": {"kind": "phase", "edge": "begin"}}}"#,
        "\n",
    );
    const STANZA_LANG_PROGRESS: &str = concat!(
        r#"{"op": "progress_v2", "event": {"scope": "load", "stage": "downloading_stanza_lang_zh", "measure": {"kind": "phase", "edge": "begin"}}}"#,
        "\n",
    );
    const HF_MODEL_PROGRESS: &str = concat!(
        r#"{"op": "progress_v2", "event": {"scope": "load", "stage": "downloading_hf_openai_whisper-large-v3", "measure": {"kind": "phase", "edge": "begin"}}}"#,
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
        r#"{"op": "capabilities", "response": {"commands": ["morphotag"], "free_threaded": false, "infer_tasks": ["morphosyntax"], "engine_versions": {"morphosyntax": null}, "stanza_catalog": {"kind": "not_installed", "import_error": "No module named 'stanza'"}}}"#,
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
                crate::worker::ResponseWait::Budget {
                    waited_for: WorkerWait::Capabilities,
                    limit: crate::worker::CAPABILITY_TIMEOUT,
                },
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
            crate::worker::ResponseWait::Budget {
                waited_for: WorkerWait::Health,
                limit: GENEROUS,
            },
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
            crate::worker::ResponseWait::Budget {
                waited_for: WorkerWait::Health,
                limit: GENEROUS,
            },
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
            crate::worker::ResponseWait::Budget {
                waited_for: WorkerWait::Health,
                limit: GENEROUS,
            },
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
            crate::worker::ResponseWait::Budget {
                waited_for: WorkerWait::Health,
                limit: GENEROUS,
            },
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
            crate::worker::ResponseWait::Budget {
                waited_for: WorkerWait::Health,
                limit: GENEROUS,
            },
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
            crate::worker::ResponseWait::Budget {
                waited_for: WorkerWait::Health,
                limit: GENEROUS,
            },
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
            crate::worker::ResponseWait::Budget {
                waited_for: WorkerWait::Health,
                limit: GENEROUS,
            },
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
            crate::worker::ResponseWait::Budget {
                waited_for: WorkerWait::Health,
                limit: GENEROUS,
            },
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
            stages.push(event.stage().to_owned());
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
            crate::worker::ResponseWait::Budget {
                waited_for: WorkerWait::Health,
                limit: GENEROUS,
            },
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
            crate::worker::ResponseWait::Budget {
                waited_for: WorkerWait::Health,
                limit: GENEROUS,
            },
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
            crate::worker::ResponseWait::Budget {
                waited_for: WorkerWait::Health,
                limit: GENEROUS,
            },
            None,
        )
        .await;
        assert!(
            matches!(result, Err(WorkerError::Protocol(_))),
            "EOF mid-preamble must error, got {result:?}"
        );
    }

    /// A budget read that never delivers a line ends at the limit with a
    /// typed timeout naming what was awaited, which the retry policy reads
    /// without parsing a message.
    #[tokio::test(start_paused = true)]
    async fn returns_a_typed_timeout_when_the_limit_passes() {
        // The writer half is held open and never written, so the read never
        // completes; paused time advances to the limit at once.
        let (reader, _writer) = tokio::io::duplex(64);
        let mut reader = BufReader::new(reader);
        let result = WorkerHandle::read_response_skipping_progress(
            &mut reader,
            crate::worker::ResponseWait::Budget {
                waited_for: WorkerWait::Capabilities,
                limit: PositiveSeconds::literal::<30>(),
            },
            None,
        )
        .await;
        match result {
            Err(WorkerError::Timeout { waited_for, limit }) => {
                assert_eq!(waited_for, WorkerWait::Capabilities);
                assert_eq!(limit.get(), 30);
            }
            other => panic!("an elapsed limit must be a typed timeout, got {other:?}"),
        }
    }

    /// One growth report of a download stage, as `_download_liveness` writes it.
    fn download_progress(grown_kib: u32) -> String {
        format!(
            "{{\"op\": \"progress_v2\", \"event\": {{\"scope\": \"load\", \"stage\": \"downloading_modelscope_iic_paraformer\", \"measure\": {{\"kind\": \"download_growth\", \"grown_kib\": {grown_kib}}}}}}}\n"
        )
    }

    /// Exercise the actual wire reader under a file observation: load
    /// growth reaches it, the final reply closes it, and request waits do
    /// not become loading operations merely because they saw a load event.
    #[tokio::test]
    async fn model_loading_progress_from_the_wire_has_an_owned_lifetime() {
        use crate::store::{ModelLoadChange, ModelLoadProgress};
        use crate::worker::pool::checkout_wait::{observing_pool_waits, tests::RecordingObserver};
        let observer = std::sync::Arc::new(RecordingObserver::default());
        let input = format!("{}{ENSURE_TASK_RESPONSE}", download_progress(4096));
        let mut reader = BufReader::new(input.as_bytes());
        let response = observing_pool_waits(
            observer.clone(),
            WorkerHandle::read_response_skipping_progress(
                &mut reader,
                crate::worker::ResponseWait::Loading {
                    waited_for: crate::worker::LoadingWait::EnsureTask {
                        task: crate::worker::InferTask::Morphosyntax,
                    },
                    stall: crate::worker::LoadingStall::new(GENEROUS),
                },
                None,
            ),
        )
        .await
        .expect("load reply");
        assert!(matches!(response, WorkerResponse::EnsureTask { .. }));
        assert_eq!(
            observer.loads.lock().unwrap().as_slice(),
            &[
                ModelLoadChange::Started,
                ModelLoadChange::Advanced(ModelLoadProgress::Downloading { grown_kib: 4096 }),
                ModelLoadChange::Ended,
            ]
        );
        observer.loads.lock().unwrap().clear();
        let mut reader = BufReader::new(input.as_bytes());
        observing_pool_waits(
            observer.clone(),
            WorkerHandle::read_response_skipping_progress(
                &mut reader,
                crate::worker::ResponseWait::Budget {
                    waited_for: WorkerWait::Health,
                    limit: GENEROUS,
                },
                None,
            ),
        )
        .await
        .expect("budget reply");
        assert!(observer.loads.lock().unwrap().is_empty());

        // A startup reports the same measured growth even when its pipe
        // then closes without a ready line; refusal closes the observation.
        let input = download_progress(8192);
        let mut reader = BufReader::new(input.as_bytes());
        let failure = observing_pool_waits(
            observer.clone(),
            WorkerHandle::read_ready_line(&mut reader, crate::worker::LoadingStall::new(GENEROUS)),
        )
        .await;
        assert!(failure.is_err());
        assert_eq!(
            observer.loads.lock().unwrap().as_slice(),
            &[
                ModelLoadChange::Started,
                ModelLoadChange::Advanced(ModelLoadProgress::Downloading { grown_kib: 8192 }),
                ModelLoadChange::Ended,
            ]
        );
    }

    /// THE 2026-11-05 CASE, as a model load. A download that keeps reporting
    /// growth is waited for however long it runs: here 20 minutes against a
    /// 120 s stall window, a report every 60 s. The fixed 120 s budget this
    /// replaced failed it at the second minute.
    #[tokio::test(start_paused = true)]
    async fn a_loading_wait_outlasts_its_window_while_progress_advances() {
        use tokio::io::AsyncWriteExt;
        let (reader, mut writer) = tokio::io::duplex(1 << 16);
        let worker = tokio::spawn(async move {
            for minute in 1..=20u32 {
                tokio::time::sleep(std::time::Duration::from_secs(60)).await;
                let _ = writer
                    .write_all(download_progress(minute * 13_000).as_bytes())
                    .await;
            }
            let _ = writer.write_all(ENSURE_TASK_RESPONSE.as_bytes()).await;
            writer
        });
        let mut reader = BufReader::new(reader);
        let result = WorkerHandle::read_response_skipping_progress(
            &mut reader,
            crate::worker::ResponseWait::Loading {
                waited_for: crate::worker::LoadingWait::EnsureTask {
                    task: crate::worker::InferTask::Asr,
                },
                stall: crate::worker::LoadingStall::new(PositiveSeconds::literal::<120>()),
            },
            None,
        )
        .await;
        assert!(
            matches!(result, Ok(WorkerResponse::EnsureTask { .. })),
            "{result:?}"
        );
        let _ = worker.await;
    }

    /// A worker cannot keep itself alive by repeating itself: the same report
    /// again is not an advance, so the window runs out from the last real one
    /// and the wait ends as `Stalled`, not as a budget timeout.
    #[tokio::test(start_paused = true)]
    async fn repeated_progress_does_not_extend_a_loading_wait() {
        use tokio::io::AsyncWriteExt;
        let (reader, mut writer) = tokio::io::duplex(1 << 16);
        let worker = tokio::spawn(async move {
            for _ in 0..10 {
                let _ = writer.write_all(download_progress(500).as_bytes()).await;
                tokio::time::sleep(std::time::Duration::from_secs(60)).await;
            }
            writer
        });
        let mut reader = BufReader::new(reader);
        let result = WorkerHandle::read_response_skipping_progress(
            &mut reader,
            crate::worker::ResponseWait::Loading {
                waited_for: crate::worker::LoadingWait::EnsureTask {
                    task: crate::worker::InferTask::Asr,
                },
                stall: crate::worker::LoadingStall::new(PositiveSeconds::literal::<120>()),
            },
            None,
        )
        .await;
        match result {
            Err(WorkerError::Stalled { waited_for, stall }) => {
                assert_eq!(
                    waited_for,
                    crate::worker::LoadingWait::EnsureTask {
                        task: crate::worker::InferTask::Asr
                    }
                );
                assert_eq!(stall.window().get(), 120);
            }
            other => panic!("a repeating worker must stall, got {other:?}"),
        }
        worker.abort();
    }

    /// Progress never extends a budget: a request that reports progress
    /// forever still ends at its limit.
    #[tokio::test(start_paused = true)]
    async fn progress_does_not_extend_a_budget() {
        use tokio::io::AsyncWriteExt;
        let (reader, mut writer) = tokio::io::duplex(1 << 16);
        let worker = tokio::spawn(async move {
            for minute in 1..=10u32 {
                let _ = writer
                    .write_all(download_progress(minute * 1_000).as_bytes())
                    .await;
                tokio::time::sleep(std::time::Duration::from_secs(10)).await;
            }
            writer
        });
        let mut reader = BufReader::new(reader);
        let result = WorkerHandle::read_response_skipping_progress(
            &mut reader,
            crate::worker::ResponseWait::Budget {
                waited_for: WorkerWait::Capabilities,
                limit: PositiveSeconds::literal::<30>(),
            },
            None,
        )
        .await;
        assert!(
            matches!(result, Err(WorkerError::Timeout { .. })),
            "{result:?}"
        );
        worker.abort();
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
            crate::worker::ResponseWait::Budget {
                waited_for: WorkerWait::Health,
                limit: GENEROUS,
            },
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
        use crate::runner::util::classify_worker_error;
        let below = format!(
            "{}{}",
            "banner\n42\n"
                .repeat((super::super::protocol::MAX_RESPONSE_STDOUT_NOISE_LINES - 1) / 2),
            "{\"op\": \"shutdown\"}\n"
        );
        let mut r = BufReader::new(below.as_bytes());
        let ok = WorkerHandle::read_response_skipping_progress(
            &mut r,
            crate::worker::ResponseWait::Budget {
                waited_for: WorkerWait::Health,
                limit: GENEROUS,
            },
            None,
        )
        .await;
        assert!(matches!(ok, Ok(WorkerResponse::Shutdown)), "{ok:?}");

        let at_limit = "banner\n".repeat(super::super::protocol::MAX_RESPONSE_STDOUT_NOISE_LINES);
        let mut r = BufReader::new(at_limit.as_bytes());
        let error = WorkerHandle::read_response_skipping_progress(
            &mut r,
            crate::worker::ResponseWait::Budget {
                waited_for: WorkerWait::Health,
                limit: GENEROUS,
            },
            None,
        )
        .await
        .expect_err("the limit closes the stream");
        assert!(
            matches!(error, WorkerError::OutputNoise { .. }),
            "{error:?}"
        );
        assert!(
            ((classify_worker_error(&error)).retryability()
                == crate::types::scheduling::Retryability::Retryable)
        );
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
