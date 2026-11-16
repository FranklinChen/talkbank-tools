//! Wire and clock obligations after operation/response correspondence is typed.

use super::*;
use crate::api::PositiveSeconds;
use crate::worker::error::WorkerWait;
use crate::worker::handle::control;
use crate::worker::{InferTask, LoadingStall};
use std::time::Duration;
use tokio::io::BufReader;

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
const ERROR_RESPONSE: &str = "{\"op\": \"error\", \"error\": \"boom\", \"kind\": \"runtime\"}\n";

#[derive(Clone, Copy, Debug)]
enum Operation {
    Health,
    Capabilities,
    Load,
    Infer,
    Batch,
}
impl Operation {
    fn terminal(self) -> &'static str {
        match self {
            Self::Health => HEALTH_RESPONSE,
            Self::Capabilities => CAPABILITIES_RESPONSE,
            Self::Load => ENSURE_TASK_RESPONSE,
            Self::Infer => INFER_RESPONSE,
            Self::Batch => BATCH_INFER_RESPONSE,
        }
    }
}

#[derive(Debug, PartialEq)]
enum Observation {
    Health,
    Capabilities,
    Loaded,
    Infer,
    Batch,
}

async fn read<R: tokio::io::AsyncBufRead + Unpin>(
    operation: Operation,
    reader: &mut R,
    progress: Option<&mpsc::Sender<ProgressEventV2>>,
) -> Result<Observation, WorkerError> {
    let mut source = LineSource(reader);
    match operation {
        Operation::Health => read_skipping_progress(&mut source, control::Health, progress)
            .await?
            .into_result()
            .map(|_| Observation::Health),
        Operation::Capabilities => {
            read_skipping_progress(&mut source, control::Capabilities::new(), progress)
                .await?
                .into_result()
                .map(|_| Observation::Capabilities)
        }
        Operation::Load => read_skipping_progress(
            &mut source,
            control::ExpectedLoad::new(
                InferTask::Morphosyntax,
                LoadingStall::new(PositiveSeconds::literal::<120>()),
            ),
            progress,
        )
        .await?
        .into_result()
        .map(|_| Observation::Loaded),
        Operation::Infer => read_skipping_progress(&mut source, control::ExpectedInfer, progress)
            .await?
            .into_result()
            .map(|_| Observation::Infer),
        Operation::Batch => {
            read_skipping_progress(&mut source, control::ExpectedBatch::new(0), progress)
                .await?
                .into_result()
                .map(|_| Observation::Batch)
        }
    }
}

/// Guards: each real control-envelope shape survives absent or stacked progress,
/// preserving forwarded request IDs, stages and completed/total values.
/// Kind: wire
/// Not a type: external JSON ordering and payload values must be decoded before
/// operation-specific completion can retain their admitted facts.
/// State: local byte readers and optional progress receivers exercise the shared reader.
/// Concurrency: each case owns its channels and has no process, filesystem or provider.
#[tokio::test]
async fn typed_control_wire_preserves_terminal_and_progress_facts() {
    let request_progress = "{\"op\":\"progress_v2\",\"event\":{\"scope\":\"request\",\"request_id\":\"req-001\",\"stage\":\"stanza_processing\",\"progress\":{\"completed\":42,\"total\":100}}}\n";
    for (operation, expected) in [
        (Operation::Health, Observation::Health),
        (Operation::Capabilities, Observation::Capabilities),
        (Operation::Load, Observation::Loaded),
        (Operation::Infer, Observation::Infer),
        (Operation::Batch, Observation::Batch),
    ] {
        for prefix in [
            String::new(),
            STANZA_CATALOG_PROGRESS.into(),
            format!(
                "{STANZA_CATALOG_PROGRESS}{STANZA_LANG_PROGRESS}{HF_MODEL_PROGRESS}{request_progress}"
            ),
        ] {
            let bytes = format!("{prefix}{}", operation.terminal());
            let mut reader = BufReader::new(bytes.as_bytes());
            let (sender, mut receiver) = mpsc::channel(prefix.lines().count().max(1));
            assert_eq!(
                read(operation, &mut reader, Some(&sender)).await.unwrap(),
                expected
            );
            let events: Vec<_> = std::iter::from_fn(|| receiver.try_recv().ok()).collect();
            assert_eq!(events.len(), prefix.lines().count());
            let actual_stages: Vec<_> = events
                .iter()
                .map(|event| serde_json::to_value(event).unwrap()["stage"].clone())
                .collect();
            let expected_stages: Vec<_> = prefix
                .lines()
                .map(|line| {
                    serde_json::from_str::<serde_json::Value>(line).unwrap()["event"]["stage"]
                        .clone()
                })
                .collect();
            assert_eq!(actual_stages, expected_stages);

            if let Some(ProgressEventV2::Request(request)) = events.last() {
                assert_eq!(request.request_id().as_ref(), "req-001");
                assert_eq!(request.stage(), "stanza_processing");
                assert_eq!(
                    (request.progress().completed(), request.progress().total()),
                    (42, 100)
                );
            }
            let mut reader = BufReader::new(bytes.as_bytes());
            assert_eq!(read(operation, &mut reader, None).await.unwrap(), expected);
        }
    }
    for operation in [
        Operation::Health,
        Operation::Capabilities,
        Operation::Load,
        Operation::Infer,
        Operation::Batch,
    ] {
        let mut reader = BufReader::new(ERROR_RESPONSE.as_bytes());
        let error = read(operation, &mut reader, None).await.unwrap_err();
        assert!(error.to_string().contains("boom"), "{error}");
    }
}

/// Guards: EOF, malformed operations, foreign task replies and noise refusal
/// retain the actual protocol or retryable output-noise reason.
/// Kind: wire
/// Not a type: external byte streams can end or name an unknown operation or
/// unrelated task before the expected terminal proof can be issued.
/// State: owned byte readers run the same decoder and typed admission as transports.
/// Concurrency: none
#[tokio::test]
async fn malformed_and_foreign_control_bytes_never_complete_an_operation() {
    for bytes in [
        format!("{STANZA_CATALOG_PROGRESS}{STANZA_LANG_PROGRESS}"),
        "{\"op\":\"unknown\"}\n".into(),
        HEALTH_RESPONSE.into(),
    ] {
        let mut reader = BufReader::new(bytes.as_bytes());
        assert!(matches!(
            read(Operation::Capabilities, &mut reader, None).await,
            Err(WorkerError::Protocol(_))
        ));
    }
    let bytes = ENSURE_TASK_RESPONSE.replace("morphosyntax", "fa");
    let mut reader = BufReader::new(bytes.as_bytes());
    assert!(matches!(
        read(Operation::Load, &mut reader, None).await,
        Err(WorkerError::Protocol(_))
    ));
    let bytes = format!(
        "{}{}",
        "banner\n42\n".repeat((super::super::protocol::MAX_RESPONSE_STDOUT_NOISE_LINES - 1) / 2),
        HEALTH_RESPONSE
    );
    assert_eq!(
        read(
            Operation::Health,
            &mut BufReader::new(bytes.as_bytes()),
            None
        )
        .await
        .unwrap(),
        Observation::Health
    );
    let bytes = "noise\n".repeat(super::super::protocol::MAX_RESPONSE_STDOUT_NOISE_LINES);
    let error = read(
        Operation::Health,
        &mut BufReader::new(bytes.as_bytes()),
        None,
    )
    .await
    .unwrap_err();
    assert!(matches!(error, WorkerError::OutputNoise { .. }));
    assert_eq!(
        crate::runner::util::classify_worker_error(&error).retryability(),
        crate::types::scheduling::Retryability::Retryable
    );
}

/// A declared stream effect owns each delayed external line. The wait-policy
/// contracts need the real clock and decoding, with no kernel IO or writer task.
struct TimedSource(std::collections::VecDeque<(Duration, String)>);
impl ResponseSource for TimedSource {
    async fn read_response(&mut self) -> Result<WorkerResponse, WorkerError> {
        match self.0.pop_front() {
            Some((delay, bytes)) => {
                tokio::time::sleep(delay).await;
                serde_json::from_str(&bytes)
                    .map_err(|error| WorkerError::Protocol(error.to_string()))
            }
            None => std::future::pending().await,
        }
    }
}

/// Guards: cold capabilities finish at 120 seconds and timeout at 301 under
/// the actual operation's fixed budget, retaining the wait kind and limit.
/// Kind: contract
/// Not a type: actual elapsed time and delayed external response arrival determine
/// whether the finite capability import budget expires.
/// State: a typed delayed-line double uses Tokio's controlled clock and real decoder.
/// Concurrency: the owned schedule and paused clock belong to this test runtime.
#[tokio::test(start_paused = true)]
async fn runtime_regression_capabilities_accept_cold_imports_and_remain_bounded() {
    enum Import {
        Arrives,
        Expires,
    }
    for case in [Import::Arrives, Import::Expires] {
        let delay = match case {
            Import::Arrives => 120,
            Import::Expires => 301,
        };
        let mut source =
            TimedSource([(Duration::from_secs(delay), CAPABILITIES_RESPONSE.into())].into());
        let result = read_skipping_progress(&mut source, control::Capabilities::new(), None).await;
        match case {
            Import::Arrives => {
                result.unwrap().into_result().unwrap();
            }
            Import::Expires => match result {
                Err(WorkerError::Timeout { waited_for, limit }) => {
                    assert_eq!(waited_for, WorkerWait::Capabilities);
                    assert_eq!(limit, crate::worker::CAPABILITY_TIMEOUT);
                }
                other => panic!(
                    "expected capability budget expiry, got {:?}",
                    other.map(|done| done.into_result())
                ),
            },
        }
    }
}

fn download_progress(grown_kib: u32) -> String {
    serde_json::json!({"op":"progress_v2","event":{"scope":"load","stage":"downloading_modelscope_iic_paraformer","measure":{"kind":"download_growth","grown_kib":grown_kib}}}).to_string() + "\n"
}

/// Guards: advancing downloads rearm loading stalls, repeated measurements do
/// not, and request budgets expire even while download progress advances.
/// Kind: contract
/// Not a type: measured progress growth and clock arrival are external facts that
/// decide rearming after an operation's wait policy has been fixed by its type.
/// State: owned delayed wire schedules exercise actual progress decoding and wait logic.
/// Concurrency: each schedule uses its own source under this runtime's paused clock.
#[tokio::test(start_paused = true)]
async fn typed_waits_distinguish_loading_progress_from_total_budgets() {
    enum Schedule {
        Advancing,
        Repeated,
        Budget,
    }
    for schedule in [Schedule::Advancing, Schedule::Repeated, Schedule::Budget] {
        let frames = (1..=20).map(|step| {
            (
                Duration::from_secs(60),
                download_progress(match schedule {
                    Schedule::Repeated => 500,
                    Schedule::Advancing | Schedule::Budget => step * 13_000,
                }),
            )
        });
        let mut source = TimedSource(
            frames
                .chain(std::iter::once((
                    Duration::ZERO,
                    ENSURE_TASK_RESPONSE.into(),
                )))
                .collect(),
        );
        match schedule {
            Schedule::Advancing => {
                read_skipping_progress(
                    &mut source,
                    control::ExpectedLoad::new(
                        InferTask::Morphosyntax,
                        LoadingStall::new(PositiveSeconds::literal::<120>()),
                    ),
                    None,
                )
                .await
                .unwrap()
                .into_result()
                .unwrap();
            }
            Schedule::Repeated => {
                let result = read_skipping_progress(
                    &mut source,
                    control::ExpectedLoad::new(
                        InferTask::Morphosyntax,
                        LoadingStall::new(PositiveSeconds::literal::<120>()),
                    ),
                    None,
                )
                .await;
                assert!(matches!(
                    result,
                    Err(WorkerError::Stalled {
                        waited_for: crate::worker::LoadingWait::EnsureTask {
                            task: InferTask::Morphosyntax
                        },
                        ..
                    })
                ));
            }
            Schedule::Budget => {
                assert!(matches!(
                    read_skipping_progress(&mut source, control::Capabilities::new(), None).await,
                    Err(WorkerError::Timeout { .. })
                ));
            }
        }
    }
}

/// Guards: decoded load progress owns Started/Advanced/Ended observations for
/// successful loading and failed startup, while a budget creates no loading lifetime.
/// Kind: contract
/// Not a type: real wire growth, reply arrival and EOF drive observable sink effects
/// after the typed wait has selected whether a loading observation exists.
/// State: the shared recording observer receives actual reader/RAII notifications.
/// Concurrency: the task-local sink and local byte streams belong to this runtime.
#[tokio::test]
async fn model_loading_progress_from_the_wire_has_an_owned_lifetime() {
    use crate::store::{ModelLoadChange, ModelLoadProgress};
    use crate::worker::pool::checkout_wait::{observing_pool_waits, tests::RecordingObserver};
    let observer = std::sync::Arc::new(RecordingObserver::default());
    let bytes = format!("{}{ENSURE_TASK_RESPONSE}", download_progress(4096));
    observing_pool_waits(
        observer.clone(),
        read(Operation::Load, &mut BufReader::new(bytes.as_bytes()), None),
    )
    .await
    .unwrap();
    assert_eq!(
        observer.loads.lock().unwrap().as_slice(),
        &[
            ModelLoadChange::Started,
            ModelLoadChange::Advanced(ModelLoadProgress::Downloading { grown_kib: 4096 }),
            ModelLoadChange::Ended
        ]
    );
    observer.loads.lock().unwrap().clear();
    let bytes = format!("{}{HEALTH_RESPONSE}", download_progress(4096));
    observing_pool_waits(
        observer.clone(),
        read(
            Operation::Health,
            &mut BufReader::new(bytes.as_bytes()),
            None,
        ),
    )
    .await
    .unwrap();
    assert!(observer.loads.lock().unwrap().is_empty());
    let bytes = download_progress(8192);
    let result = observing_pool_waits(
        observer.clone(),
        WorkerHandle::read_ready_line(
            &mut BufReader::new(bytes.as_bytes()),
            LoadingStall::new(PositiveSeconds::literal::<10>()),
        ),
    )
    .await;
    assert!(result.is_err());
    assert_eq!(
        observer.loads.lock().unwrap().as_slice(),
        &[
            ModelLoadChange::Started,
            ModelLoadChange::Advanced(ModelLoadProgress::Downloading { grown_kib: 8192 }),
            ModelLoadChange::Ended
        ]
    );
}
