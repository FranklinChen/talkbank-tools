use super::*;
use crate::api::MemoryMb;
use crate::worker::pool::task_load::tests::{
    coref_request, declared_invocation, ready, test_admission,
};
use tokio::io::{AsyncBufReadExt, BufReader};

/// Guards: submitted work retains its actual batch lease and executor slot after
/// caller cancellation, releasing both when its reply stream retires.
/// Kind: contract
/// Not a type: caller cancellation and stream closure are external asynchronous
/// effects whose actual resource release order still needs observation.
/// State: an isolated memory ledger measures the real coreference batch lease,
/// and duplex pipes exercise the production write and reader tasks without models.
/// Concurrency: local pipe halves, ledger and one channel belong to this runtime.
#[tokio::test(start_paused = true)]
async fn submitted_work_retains_capacity_after_its_caller_ends() {
    use crate::worker::pool::job_tracker::{CURRENT_JOB_ID, JobWorkerTracker, TrackerGuard};
    enum CallerExit {
        Cancelled,
        TimedOut,
    }
    for end in [CallerExit::Cancelled, CallerExit::TimedOut] {
        let (_directory, admission) = test_admission();
        let request = coref_request();
        let permit = declared_invocation(ready(&request), &admission).await;
        let reserved = || admission.reserved_for_test();
        let held = reserved();
        assert!(held.0 > 0);
        let tracker = JobWorkerTracker::default();
        let job = crate::api::JobId::from("pending-contract");
        let pid = WorkerPid(std::process::id());
        let registration = CURRENT_JOB_ID
            .scope(job.clone(), async { TrackerGuard::new(&tracker, pid) })
            .await;
        let (requests, worker_requests) = tokio::io::duplex(4096);
        let (mut worker_responses, responses) = tokio::io::duplex(4096);
        let channel = SharedGpuChannel::open(
            BufReader::new(responses),
            requests,
            pid,
            TaskTimeoutOverrides::NONE,
            1,
        );
        let mut worker_requests = BufReader::new(worker_requests);
        let mut line = String::new();
        let mut execution = Box::pin(channel.execute_admitted(permit, registration));
        tokio::select! {
            result = &mut execution => panic!("worker has not replied: {result:?}"),
            read = worker_requests.read_line(&mut line) => assert!(read.unwrap() > 0),
        }
        assert!(line.contains("execute_v2"));
        match end {
            CallerExit::Cancelled => drop(execution),
            CallerExit::TimedOut => {
                assert!(matches!(execution.await, Err(WorkerError::Timeout { .. })));
            }
        }
        assert_eq!(reserved(), held, "submitted work retains batch memory");
        assert_eq!(channel.dispatch_semaphore.available_permits(), 0);
        assert_eq!(
            tracker.snapshot(&job),
            vec![pid],
            "submitted work retains its cancellation target"
        );
        // Rejoin before load preparation or capacity admission. The original
        // pending owner already owns both, so no work or lease is duplicated.
        let preparation = crate::worker::pool::LoadPreparation {
            admission: admission.clone(),
            stall: crate::worker::LoadingStall::DEFAULT_ENSURE_TASK,
        };
        let registration = TrackerGuard::new(&tracker, pid);
        let mut rejoined =
            Box::pin(channel.execute_v2(ready(&request), &preparation, registration));
        assert!(futures::poll!(&mut rejoined).is_pending());
        assert_eq!(reserved(), held);
        assert_eq!(channel.dispatch_semaphore.available_permits(), 0);
        match end {
            CallerExit::Cancelled => {
                let response = serde_json::json!({"op":"execute_v2","response":{
                    "request_id":request.request_id,"outcome":{"kind":"error","code":"invalid_payload","message":"declared refusal"},"elapsed_s":0.0
                }});
                worker_responses
                    .write_all(format!("{response}\n").as_bytes())
                    .await
                    .unwrap();
                assert_eq!(rejoined.await.unwrap().request_id(), &request.request_id);
            }
            CallerExit::TimedOut => {
                drop(worker_responses);
                assert!(matches!(
                    rejoined.await,
                    Err(WorkerError::ProcessExited { .. })
                ));
            }
        }
        assert_eq!(reserved(), MemoryMb(0));
        assert_eq!(channel.dispatch_semaphore.available_permits(), 1);
        assert!(tracker.snapshot(&job).is_empty());
        line.clear();
        assert!(
            futures::poll!(Box::pin(worker_requests.read_line(&mut line))).is_pending(),
            "rejoining must not write another packet"
        );
    }
}

/// Guards: cancelling an incomplete real write retires its pending ownership
/// and stream before another invocation can reuse the transport.
/// Kind: contract
/// Not a type: an AsyncWrite can accept a prefix and then suspend, so actual
/// partial bytes and future cancellation determine transport uncertainty.
/// State: a one-byte duplex pipe forces partial output with a real batch lease.
/// Concurrency: the test owns the channel, both pipe halves and isolated ledger.
#[tokio::test]
async fn cancelling_a_partial_write_retires_the_stream_and_its_owners() {
    use crate::worker::pool::job_tracker::{JobWorkerTracker, TrackerGuard};
    let (_directory, admission) = test_admission();
    let request = coref_request();
    let permit = declared_invocation(ready(&request), &admission).await;
    let (requests, _worker_requests) = tokio::io::duplex(1);
    let (_worker_responses, responses) = tokio::io::duplex(1);
    let pid = WorkerPid(std::process::id());
    let channel = SharedGpuChannel::open(
        BufReader::new(responses),
        requests,
        pid,
        TaskTimeoutOverrides::NONE,
        1,
    );
    let registration = TrackerGuard::new(&JobWorkerTracker::default(), pid);
    let mut writing = Box::pin(channel.execute_admitted(permit, registration));
    assert!(futures::poll!(&mut writing).is_pending());
    assert_eq!(channel.routes.pending.waiting(), 1);
    assert!(admission.reserved_for_test().0 > 0);
    drop(writing);
    assert!(channel.check_available().is_err());
    assert_eq!(channel.routes.pending.waiting(), 0);
    assert_eq!(admission.reserved_for_test(), MemoryMb(0));
    assert_eq!(channel.dispatch_semaphore.available_permits(), 1);
}

/// Guards: stdio and TCP reject foreign request IDs and same-ID results for
/// another task before attributing terminal bytes to the original invocation.
/// Kind: contract
/// Not a type: external workers can write well-formed foreign replies, so
/// terminal admission must bind their actual identity and result task.
/// State: a shell subprocess and loopback connection exercise both real adapters
/// with the sealed declared-load fixture and isolated memory ledger.
/// Concurrency: each adapter owns its child or OS-assigned port and socket halves.
#[tokio::test]
async fn sequential_reply_must_name_its_actual_invocation() {
    use crate::worker::tcp_handle::{TcpWorkerHandle, TcpWorkerInfo};
    enum Adapter {
        Stdio,
        Tcp,
    }
    let request = coref_request();
    for response in [
        serde_json::json!({"request_id":"foreign-request","outcome":{"kind":"error","code":"invalid_payload","message":"foreign"},"elapsed_s":0.0}),
        serde_json::json!({"request_id":request.request_id,"outcome":{"kind":"success"},"result":{"kind":"utseg_result","items":[]},"elapsed_s":0.0}),
    ] {
        for adapter in [Adapter::Stdio, Adapter::Tcp] {
            let (_directory, admission) = test_admission();
            let permit = declared_invocation(ready(&request), &admission).await;
            let terminal = format!(
                "{}\n",
                serde_json::json!({"op":"execute_v2","response":response})
            );
            let result = match adapter {
                Adapter::Stdio => {
                    let ready_line = crate::worker::handle::test_support::ready_line();
                    let script = format!(
                        "printf '%s\\n' '{}'; read request; printf '%s\\n' '{}'",
                        ready_line.trim(),
                        terminal.trim()
                    );
                    let mut worker =
                        crate::worker::handle::WorkerHandle::spawn_stub_for_test(&script).await;
                    worker.execute_admitted(permit, None).await
                }
                Adapter::Tcp => {
                    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
                    let info = TcpWorkerInfo {
                        host: "127.0.0.1".into(),
                        port: listener.local_addr().unwrap().port(),
                        profile: crate::worker::WorkerProfile::Stanza,
                        lang: crate::api::LanguageCode3::eng().into(),
                        engine_overrides: String::new(),
                        pid: WorkerPid(std::process::id()),
                        task_timeouts: TaskTimeoutOverrides::NONE,
                    };
                    let mut worker = TcpWorkerHandle::connect(info).await.unwrap();
                    let reply = async {
                        let (socket, _) = listener.accept().await.unwrap();
                        let (reader, mut writer) = tokio::io::split(socket);
                        let mut reader = BufReader::new(reader);
                        let mut line = String::new();
                        assert!(reader.read_line(&mut line).await.unwrap() > 0);
                        writer.write_all(terminal.as_bytes()).await.unwrap();
                    };
                    let (result, ()) = tokio::join!(worker.execute_admitted(permit, None), reply);
                    result
                }
            };
            assert!(
                matches!(result, Err(WorkerError::Protocol(_))),
                "{result:?}"
            );
            assert_eq!(admission.reserved_for_test(), MemoryMb(0));
        }
    }
}
