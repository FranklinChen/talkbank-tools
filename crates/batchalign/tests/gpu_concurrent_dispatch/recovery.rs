//! A dead shared process must not remain the pool's permanent worker.

use super::*;
use batchalign::worker::pool::WorkerTransport;
use std::sync::Arc;

/// Guards: After worker EOF, concurrent callers complete through a single live replacement.
/// Kind: journey
/// Not a type: Only the process and its pipe can reveal when the reader observes an actual child exit.
/// State: This test signals the pool-owned stdio child and verifies both process identities leave the OS.
/// Concurrency: Four callers race for replacement after the initial dispatch observes the crash.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn next_dispatches_replace_a_dead_shared_worker_once() {
    let python = super::worker_python();
    let pool = Arc::new(test_pool(python));
    pool.dispatch_gpu_for_test(
        &GPU_PLAN,
        &LanguageCode3::eng(),
        &gpu_execute_request("before-crash"),
    )
    .await
    .expect("warm the actual shared stdio worker");
    let original_pid = pool
        .worker_summary_entries()
        .await
        .into_iter()
        .find(|entry| entry.concurrent && entry.transport == WorkerTransport::Stdio)
        .expect("this pool owns a shared stdio worker")
        .pid;

    // This PID belongs to the live child just spawned by this pool.
    let original = i32::try_from(original_pid.0).expect("a Unix PID fits pid_t");
    nix::sys::signal::kill(
        nix::unistd::Pid::from_raw(original),
        nix::sys::signal::Signal::SIGKILL,
    )
    .expect("signal the shared worker");

    // A request racing the reader's discovery of EOF may fail. It must not
    // poison every later request with the same dead OnceCell occupant.
    let _ = tokio::time::timeout(
        Duration::from_secs(30),
        pool.dispatch_gpu_for_test(
            &GPU_PLAN,
            &LanguageCode3::eng(),
            &gpu_execute_request("observe-crash"),
        ),
    )
    .await
    .expect("a dead worker must fail promptly");

    let mut callers = Vec::new();
    for index in 0..4 {
        let pool = pool.clone();
        callers.push(tokio::spawn(async move {
            let request = gpu_execute_request(&format!("after-crash-{index}"));
            tokio::time::timeout(
                Duration::from_secs(30),
                pool.dispatch_gpu_for_test(&GPU_PLAN, &LanguageCode3::eng(), &request),
            )
            .await
        }));
    }
    let mut outcomes = Vec::new();
    for caller in callers {
        outcomes.push(caller.await.expect("dispatch task panicked"));
    }
    let replacements = pool.worker_summary_entries().await;
    pool.shutdown().await;

    for outcome in outcomes {
        outcome
            .expect("replacement must not hang")
            .expect("later requests must use a fresh live worker");
    }
    assert_eq!(
        replacements.len(),
        1,
        "one replacement is shared by all callers"
    );
    assert_ne!(replacements[0].pid, original_pid);
    wait_for_process_exit(original_pid.0).await;
    wait_for_process_exit(replacements[0].pid.0).await;
}
