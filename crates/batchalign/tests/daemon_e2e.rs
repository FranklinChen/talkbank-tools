//! Daemon lifecycle E2E test.
//!
//! Verifies that `batchalign3 serve start` / `serve status` / `serve stop`
//! work as subprocesses with process isolation.
//!
//! Run: `cargo test -p batchalign --test cli_integration_suite daemon_e2e::`
// Integration tests are exempt from the crate's deny-level panic lints,
// matching the src/lib.rs `#![cfg_attr(test, allow(...))]` pattern
// (see docs/panic-audit/).
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable,
    clippy::todo,
    clippy::unimplemented
)]

use crate::cli_common;
use crate::live_deadline::CliRunBudget;

use predicates::prelude::*;

use cli_common::CliHarness;

/// Start a server in the background, check its status, then stop it.
///
/// Uses a fresh HOME tempdir and random port for isolation.
#[test]
fn daemon_lifecycle_start_status_stop() {
    let python = crate::prerequisite::PythonWorker::require().interpreter();

    let harness = CliHarness::new();

    // Write a minimal server.yaml with a random port
    // Use port 0 to let the OS assign, but serve start needs a concrete port.
    // Pick a high random port unlikely to conflict.
    let port: u16 = 19000 + (std::process::id() as u16 % 1000);
    let config = format!("host: 127.0.0.1\nport: {port}\nauto_daemon: false\n");
    std::fs::write(harness.server_config_path(), &config).unwrap();

    // Start server in background (not --foreground)
    let start_result = harness
        .cmd()
        .args([
            "serve",
            "start",
            "--test-echo",
            "--python",
            &python,
            "--port",
            &port.to_string(),
            "--config",
            harness.server_config_path().to_str().unwrap(),
        ])
        .timeout(CliRunBudget::DaemonStart.as_duration())
        .ok();

    // A server that does not start is this test's failure, not a skip.
    match start_result {
        Ok(output) if !output.status.success() => {
            panic!("serve start failed: {output:?}")
        }
        Err(e) => panic!("serve start errored: {e}"),
        _ => {}
    }

    // Give it a moment to actually start
    std::thread::sleep(std::time::Duration::from_secs(3));

    // Check status
    harness
        .cmd()
        .args([
            "serve",
            "status",
            "--server",
            &format!("http://127.0.0.1:{port}"),
        ])
        .timeout(CliRunBudget::ServerRoundTrip.as_duration())
        .assert()
        .success()
        .stderr(predicate::str::contains("Status:").or(predicate::str::contains("cannot reach")));

    // Stop
    harness
        .cmd()
        .args(["serve", "stop"])
        .timeout(CliRunBudget::ServeStop.as_duration())
        .assert()
        .success();

    // Verify status shows unreachable after stop. No sleep: `serve stop`
    // returns only after the server process has exited.
    harness
        .cmd()
        .args([
            "serve",
            "status",
            "--server",
            &format!("http://127.0.0.1:{port}"),
        ])
        .timeout(CliRunBudget::ServerRoundTrip.as_duration())
        .assert()
        .success()
        .stderr(predicate::str::contains("cannot reach"));
}
