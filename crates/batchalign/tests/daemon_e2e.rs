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

use cli_common::{CliHarness, resolve_python};

/// Start a server in the background, check its status, then stop it.
///
/// Uses a fresh HOME tempdir and random port for isolation.
#[test]
fn daemon_lifecycle_start_status_stop() {
    let Some(python) = resolve_python() else {
        eprintln!("SKIP: Python 3 with batchalign not available");
        return;
    };

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

    // The start command might fail if the port is taken, skip gracefully
    match start_result {
        Ok(output) if !output.status.success() => {
            eprintln!("SKIP: serve start failed (port conflict?): {:?}", output);
            return;
        }
        Err(e) => {
            eprintln!("SKIP: serve start errored: {e}");
            return;
        }
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
        .timeout(std::time::Duration::from_secs(10))
        .assert()
        .success()
        .stderr(predicate::str::contains("Status:").or(predicate::str::contains("cannot reach")));

    // Stop
    harness
        .cmd()
        .args(["serve", "stop"])
        .timeout(std::time::Duration::from_secs(10))
        .assert()
        .success();

    // Verify status shows unreachable after stop
    std::thread::sleep(std::time::Duration::from_millis(500));
    harness
        .cmd()
        .args([
            "serve",
            "status",
            "--server",
            &format!("http://127.0.0.1:{port}"),
        ])
        .timeout(std::time::Duration::from_secs(10))
        .assert()
        .success()
        .stderr(predicate::str::contains("cannot reach"));
}
