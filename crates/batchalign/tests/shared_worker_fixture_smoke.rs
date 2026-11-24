//! Real stdio startup and reuse through the shared echo fixtures.

use crate::common::echo_worker::EchoWorker;
use batchalign::api::{LanguageCode3, WorkerLanguage};
use batchalign::worker::WorkerHealthStatus;

/// Guards: Two equivalent checkouts retain the original child across caller
/// reactors. Profile and task bootstrap produce distinct, correctly labelled
/// workers, each carrying the ready payload's Python runtime identity.
/// Kind: contract
/// Not a type: The choices of launch and slot are closed in Rust. The kernel
/// assigns a process to the pipe and Python emits the ready and health bytes.
/// Reactor lifetime affects whether those real pipes still respond.
/// State: Two actual echo workers exercise profile and task launch without
/// models, provider accounts or TCP registration.
/// Concurrency: The first caller reactor ends before the second checkout.
/// Each lease exclusively holds its pipe. The pool retains the child reactor.
#[test]
fn echo_checkout_keeps_the_child_alive_across_caller_reactors() {
    let caller = || {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("caller runtime")
    };
    let profile_pid = caller().block_on(async {
        let lease = EchoWorker::Profile
            .checkout()
            .await
            .expect("profile starts");
        let identity = lease.runtime_identity();
        assert!(!identity.python_version().is_empty());
        assert_eq!(identity.python_executable_sha256().as_str().len(), 64);
        lease.pid()
    });

    caller().block_on(async {
        let (profile, task) = tokio::join!(
            EchoWorker::Profile.checkout(),
            EchoWorker::Morphosyntax.checkout()
        );
        let mut profile = profile.expect("profile remains available");
        let mut task = task.expect("task starts");
        assert_eq!(profile.pid(), profile_pid);
        assert_ne!(profile.pid(), task.pid());

        for (lease, command) in [
            (&mut profile, "profile:stanza"),
            (&mut task, "infer:morphosyntax"),
        ] {
            let health = lease.health_check().await.expect("child replies");
            assert_eq!(health.status, WorkerHealthStatus::Ok);
            assert_eq!(health.command, command);
            assert_eq!(health.lang, WorkerLanguage::from(LanguageCode3::eng()));
            assert_eq!(health.pid, lease.pid());
        }
    });
}
