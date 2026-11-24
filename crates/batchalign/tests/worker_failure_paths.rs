// Integration test target: Cargo compiles this as a separate crate,
// so the lib's `cfg_attr(test, ...)` allow does not apply. Test code
// uses `unwrap`/`expect` by convention.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable
)]

//! Worker failure-path integration tests (T077, T078, T079).
//!
//! These tests verify:
//! - T077: Worker crash propagates errors correctly to callers
//! - T078: Worker ready timeout fires when worker is too slow to start
//! - T079: Multi-file jobs preserve completed results when later files fail
//!
//! All tests use `--test-echo` workers (no ML models). Failures are injected
//! by killing worker processes (SIGKILL) or using very short timeouts.

use std::collections::BTreeMap;
use std::time::Duration;

use crate::live_deadline::{ProgressSnapshot, ServerTestDeadline, WaitSubject};
use batchalign::api::{LanguageCode3, ReleasedCommand, WorkerLanguage};
use batchalign::host_facts::PerProfile;
use batchalign::worker::handle::{WorkerConfig, WorkerHandle};
use batchalign::worker::pool::{PoolConfig, WorkerPool};
use batchalign::worker::{BatchInferRequest, InferTask, WorkerProfile};
use serde_json::{Value, json};

fn batch_request(task: InferTask, items: Vec<Value>) -> BatchInferRequest {
    BatchInferRequest {
        task,
        lang: LanguageCode3::eng(),
        items,
        mwt: BTreeMap::new(),
    }
}

fn test_pool(python: String) -> WorkerPool {
    WorkerPool::new(PoolConfig {
        health_check_interval_s: batchalign::api::PositiveSeconds::literal::<600>(), // disable periodic health checks
        startup_stall: batchalign::worker::LoadingStall::new(
            batchalign::api::PositiveSeconds::literal::<30>(),
        ),
        test_echo: true,
        max_workers_per_key: PerProfile::uniform(8),
        verbose: 0,
        ..PoolConfig::for_tests_with_python(python)
    })
}

// ---------------------------------------------------------------------------
// T077: Worker crash/restart/retry failure-path tests
// ---------------------------------------------------------------------------

/// Guards: Losing a native worker after it reads a batch returns an error.
/// Kind: contract
/// Not a type: A submitted invocation can lose its real pipe after the other
/// process reads the request but before the response exists.
/// State: A shell worker emits the ready frame, acknowledges reading the batch
/// by creating a private marker, and then blocks on its own stdin.
/// Concurrency: The signal is gated on the read marker and targets that child.
#[cfg(unix)]
#[tokio::test]
async fn worker_killed_mid_batch_infer_returns_error() {
    use std::os::unix::fs::PermissionsExt;

    let dir = tempfile::tempdir().expect("private shell worker");
    let executable = dir.path().join("worker");
    let marker = dir.path().join("batch-active");
    let ready = crate::worker_integration::synthetic_runtime_ready(1234);
    let (prefix, suffix) = ready.split_once("\"pid\":1234").expect("ready PID field");
    std::fs::write(
        &executable,
        format!(
            r#"#!/bin/sh
printf '%s\n' '{prefix}"pid":'"$$"'{suffix}'
IFS= read -r request || exit 1
case "$request" in
    *batch_infer*) : > "$(dirname "$0")/batch-active" ;;
    *) exit 2 ;;
esac
IFS= read -r unused
"#,
        ),
    )
    .expect("write shell worker");
    std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o755)).expect("chmod");
    let config = WorkerConfig {
        test_echo: true,
        ..WorkerConfig::for_tests_with_python(executable.to_string_lossy().into_owned())
    };
    let mut handle = WorkerHandle::spawn(
        config,
        std::sync::Arc::new(batchalign::host_memory::NoIdleWorkers),
    )
    .await
    .expect("spawn");
    let child = FixturePid::admit(handle.pid());
    let request = batch_request(InferTask::Morphosyntax, vec![json!({"words": ["hello"]})]);
    let terminate = async {
        tokio::time::timeout(Duration::from_secs(10), async {
            while !marker.is_file() {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("the child never read the batch request");
        child.kill();
    };

    let (result, ()) = tokio::join!(handle.batch_infer(&request), terminate);
    assert!(
        result.is_err(),
        "a killed worker cannot complete the request"
    );
}

/// Guards: One group's crashed process is replaced and a live sibling group
/// keeps its original process and answers correctly.
/// Kind: journey
/// Not a type: A native exit can invalidate a checked-in worker while other
/// groups remain usable. The kernel and real pipes witness recovery.
/// State: Two private live echo groups, one signal and typed pool snapshots.
/// Concurrency: Wait for the victim to exit before the replacement dispatch.
#[cfg(unix)]
#[tokio::test]
async fn pool_recovers_after_worker_crash() {
    let pool = test_pool(crate::prerequisite::worker_test_python());
    let lang = LanguageCode3::eng();
    let dispatch = |task, marker: &'static str| {
        let pool = &pool;
        let lang = &lang;
        async move {
            let item = json!({"probe": marker});
            let result = pool
                .dispatch_batch_infer(lang, &batch_request(task, vec![item.clone()]))
                .await
                .expect("echo dispatch");
            assert_eq!(result.results[0].outcome.result(), Some(&item));
            let label = WorkerProfile::for_task(task).label();
            pool.worker_summary_entries()
                .await
                .into_iter()
                .find(|entry| entry.profile == label)
                .expect("idle worker for the task's profile")
                .pid
        }
    };
    let victim = dispatch(InferTask::Morphosyntax, "initial").await;
    let other = dispatch(InferTask::Translate, "other-initial").await;
    let child = FixturePid::admit(victim);
    child.kill();
    child.wait_for_exit().await;

    assert_eq!(
        dispatch(InferTask::Translate, "other-still-serving").await,
        other
    );
    assert_ne!(dispatch(InferTask::Morphosyntax, "recovered").await, victim);
    pool.shutdown().await;
}

/// Only an admitted positive native PID can be signaled in this fixture.
#[cfg(unix)]
struct FixturePid(std::num::NonZeroI32);

#[cfg(unix)]
impl FixturePid {
    fn admit(pid: batchalign::worker::WorkerPid) -> Self {
        Self(
            std::num::NonZeroI32::new(i32::try_from(*pid).expect("Unix PID"))
                .expect("live worker must have a positive PID"),
        )
    }

    fn kill(&self) {
        nix::sys::signal::kill(
            nix::unistd::Pid::from_raw(self.0.get()),
            nix::sys::signal::Signal::SIGKILL,
        )
        .expect("signal the worker");
    }

    async fn wait_for_exit(self) {
        let observe = batchalign::blocking::spawn_in_span(move || {
            // Observe exit without taking the pool's right to reap its child.
            let mut info = std::mem::MaybeUninit::uninit();
            loop {
                // Reason: `nix` has no `waitid` on macOS, and only `waitid`
                // with `WNOWAIT` observes an exit without reaping the child.
                #[allow(unsafe_code)]
                // SAFETY: `info` is a valid out-pointer for one `siginfo_t`.
                let status = unsafe {
                    libc::waitid(
                        libc::P_PID,
                        self.0.get() as libc::id_t,
                        info.as_mut_ptr(),
                        libc::WEXITED | libc::WNOWAIT,
                    )
                };
                if status == 0 {
                    return;
                }
                let error = std::io::Error::last_os_error();
                match error.raw_os_error() {
                    Some(libc::ECHILD) => return,
                    Some(libc::EINTR) => continue,
                    _ => panic!("observe worker exit: {error}"),
                }
            }
        });
        tokio::time::timeout(Duration::from_secs(10), observe)
            .await
            .expect("worker stayed alive after SIGKILL")
            .expect("join exit observation");
    }
}

/// Guards: The missing-ready error includes the real stderr traceback despite startup progress.
/// Kind: wire
/// Not a type: The child's diagnostic and readiness verdict cross independent OS streams and can arrive separately.
/// State: An executable shell emits advance progress and stderr, then exits without ready.
/// Concurrency: The test owns the child and both captured streams.
#[cfg(unix)]
#[tokio::test]
async fn worker_death_before_ready_surfaces_stderr() {
    use std::io::Write;
    use std::os::unix::fs::PermissionsExt;

    let tmp = tempfile::TempDir::new().expect("tempdir");
    let script_path = tmp.path().join("crashing-python");
    {
        let mut f = std::fs::File::create(&script_path).expect("create script");
        writeln!(f, "#!/bin/sh").expect("write");
        writeln!(
            f,
            "echo '{{\"op\": \"progress_v2\", \"event\": {{\"scope\": \"load\", \"stage\": \"refreshing_stanza_catalog\", \"measure\": {{\"kind\": \"phase\", \"edge\": \"begin\"}}}}}}'"
        )
        .expect("write");
        writeln!(f, "echo 'Traceback (most recent call last):' >&2").expect("write");
        writeln!(
            f,
            "echo 'RuntimeError: synthetic bootstrap crash for test' >&2"
        )
        .expect("write");
        writeln!(f, "exit 3").expect("write");
    }
    let mut perms = std::fs::metadata(&script_path)
        .expect("stat script")
        .permissions();
    perms.set_mode(0o755);
    std::fs::set_permissions(&script_path, perms).expect("chmod script");

    let config = WorkerConfig {
        test_echo: true,
        profile: WorkerProfile::Stanza,
        lang: WorkerLanguage::from(LanguageCode3::eng()),
        startup_stall: batchalign::worker::LoadingStall::new(
            batchalign::api::PositiveSeconds::literal::<30>(),
        ),
        ..WorkerConfig::for_tests_with_python(script_path.to_string_lossy().into_owned())
    };

    let err = match WorkerHandle::spawn(
        config,
        std::sync::Arc::new(batchalign::host_memory::NoIdleWorkers),
    )
    .await
    {
        Err(err) => err,
        Ok(_) => panic!("spawn must fail when the worker dies before ready"),
    };
    let batchalign::worker::error::WorkerError::ReadyParseFailed(message) = err else {
        panic!("expected a missing-ready failure, got {err}");
    };
    assert!(message.contains("worker closed stdout without emitting ready signal"));
    assert!(message.contains("RuntimeError: synthetic bootstrap crash for test"));
}

// ---------------------------------------------------------------------------
// T078: Worker timeout integration tests
// ---------------------------------------------------------------------------

/// Guards: A multi-file job serves each completed file both in the full
/// results response and individually by its filename.
/// Kind: journey
/// Not a type: The HTTP job runner and its result store must preserve names
/// and content across real submission, completion and retrieval.
/// State: One isolated HTTP session and three valid synthetic CHAT files.
/// Concurrency: The fixture leases its server and owns storage until completion.
#[tokio::test]
async fn multi_file_job_produces_per_file_results() {
    use batchalign::api::{
        FilePayload, JobInfo, JobResultResponse, JobStatus, JobSubmission, LanguageSpec,
        NumSpeakers,
    };
    use batchalign::options::{CommandOptions, CommonOptions, MorphotagOptions};

    let session = crate::common::test_server_fixture::acquire_test_server_session().await;
    let base_url = session.base_url();
    let client = session.client();

    // Submit a job with 3 files. Morphotag requires
    // `LanguageSpec::PerFile`; each file carries `@Languages: eng`
    // so per-file resolution succeeds at dispatch time.
    let submission = JobSubmission {
        command: ReleasedCommand::Morphotag,
        lang: LanguageSpec::PerFile,
        num_speakers: NumSpeakers(1),
        files: vec![
            FilePayload {
                filename: "file_a.cha".into(),
                content: "@UTF8\n@Languages:\teng\n@Begin\n*CHI:\thello .\n@End\n".into(),
            },
            FilePayload {
                filename: "file_b.cha".into(),
                content: "@UTF8\n@Languages:\teng\n@Begin\n*CHI:\tworld .\n@End\n".into(),
            },
            FilePayload {
                filename: "file_c.cha".into(),
                content: "@UTF8\n@Languages:\teng\n@Begin\n*CHI:\tgoodbye .\n@End\n".into(),
            },
        ],
        media_files: vec![],
        media_mapping: Default::default(),
        media_subdir: Default::default(),
        source_dir: Default::default(),
        options: CommandOptions::Morphotag(MorphotagOptions {
            common: CommonOptions::default(),

            ..Default::default()
        }),
        paths_mode: false,
        source_paths: vec![],
        output_paths: vec![],
        display_names: vec![],
        debug_traces: false,
        before_paths: vec![],
    };

    let resp = client
        .post(format!("{base_url}/jobs"))
        .json(&submission)
        .send()
        .await
        .expect("POST /jobs");
    assert_eq!(resp.status(), 200);
    let info: JobInfo = resp.json().await.expect("parse");
    assert_eq!(info.total_files, 3);

    // Poll until complete.
    let mut deadline = ServerTestDeadline::new(WaitSubject::job_completion(&info.job_id));
    loop {
        let resp = client
            .get(format!("{base_url}/jobs/{}", info.job_id))
            .send()
            .await
            .expect("GET job");
        let job: JobInfo = resp.json().await.expect("parse");
        if matches!(
            job.status,
            JobStatus::Completed | JobStatus::Failed | JobStatus::Cancelled
        ) {
            break;
        }
        deadline.keep_waiting(ProgressSnapshot::job(&job)).await;
    }

    // Verify all 3 files have individual results.
    let resp = client
        .get(format!("{base_url}/jobs/{}/results", info.job_id))
        .send()
        .await
        .expect("GET results");
    let results: JobResultResponse = resp.json().await.expect("parse results");
    assert_eq!(
        results.files.len(),
        3,
        "should have per-file results for all 3 files"
    );

    // Each file should have content and no error (test-echo processes all).
    for (i, file_result) in results.files.iter().enumerate() {
        assert!(
            file_result.error.is_none(),
            "file {i} ({}) should have no error",
            file_result.filename
        );
        assert!(
            !file_result.content.is_empty(),
            "file {i} ({}) should have content",
            file_result.filename
        );
    }

    // Verify individual file retrieval works.
    for filename in &["file_a.cha", "file_b.cha", "file_c.cha"] {
        let resp = client
            .get(format!(
                "{base_url}/jobs/{}/results/{filename}",
                info.job_id
            ))
            .send()
            .await
            .expect("GET single file result");
        assert_eq!(
            resp.status(),
            200,
            "individual file {filename} should be retrievable"
        );
    }
    session.close().await;
}
