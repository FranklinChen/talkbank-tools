//! Binary subprocess tests for `batchalign3`.
//!
//! Uses `assert_cmd` to run the binary and verify exit codes, stdout, stderr.
//! Argument-only tests use the shared baseline environment. Processing tests
//! own a CliHarness with isolated HOME and explicit setup configuration; live
//! dispatch tests additionally own their local server session.
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

use crate::{cli_common, common};

use predicates::prelude::*;

use batchalign::media::tools::MediaTool;
use cli_common::{
    CliHarness, MINIMAL_CHAT, cli_cmd as cmd, resolve_python, start_live_server,
    transcode_audio_to_mp4, write_silent_mp4, write_silent_wav,
};
use common::test_server_fixture::acquire_test_server_session;

// ---------------------------------------------------------------------------
// Version / help
// ---------------------------------------------------------------------------

/// The port the auto-spawned daemon published for itself.
///
/// Used by the auto-daemon tests, which never run `serve start` and so cannot
/// take the port from [`start_daemon_on_ephemeral_port`]. Both slots are
/// checked because `transcribe` may be routed to either the main daemon or the
/// transcribe sidecar.
///
/// This replaces `reserve_ephemeral_port`, which bound a port only to learn its
/// number, released it, and hoped the daemon won the ensuing race. It lost that
/// race twice in one session under full-suite load. Nothing reserves a port
/// now: the daemon binds whatever the OS gives it and reports what it got.
fn published_daemon_port(harness: &CliHarness) -> u16 {
    use batchalign::server_handshake::{HandshakeSlot, ServerHandshake};
    for slot in [HandshakeSlot::Main, HandshakeSlot::Sidecar] {
        if let Ok(Some(handshake)) = ServerHandshake::read(harness.state_dir(), slot)
            && let Some(port) = handshake.bound_port()
        {
            return port.get();
        }
    }
    panic!("the auto-spawned daemon published no port in either slot");
}

/// Start the test daemon on an OS-chosen port and return the port it bound.
///
/// No port number is reserved, predicted or handed over: the config asks for
/// `port: 0`, the daemon binds whatever the OS gives it, and this reads back
/// the port it published. There is no window for another process to take the
/// port because no port was claimed in advance, so unlike the reserve-then-bind
/// helper this replaced, there is nothing to retry.
///
/// The `extra_config` lines are appended to the host/port pair.
fn start_daemon_on_ephemeral_port(
    harness: &CliHarness,
    python_path: &str,
    extra_config: &str,
) -> u16 {
    harness.write_server_config(&format!("host: 127.0.0.1\nport: 0\n{extra_config}"));

    let start = harness
        .cmd()
        .env("BATCHALIGN_PYTHON", python_path)
        .args(["serve", "start", "--test-echo"])
        .timeout(std::time::Duration::from_secs(60))
        .output()
        .expect("start CLI test server");
    // The full evidence, because the parent's stderr ALONE cannot explain a
    // failure here. `serve start` redirects the CHILD daemon's stderr into
    // `server.log`, so a child that dies leaves the parent with nothing to
    // print; and when this command hits the 60 s timeout above, assert_cmd
    // kills it, leaving only whatever it had written before the wait. That is
    // exactly what one intermittent failure on 2026-08-27 looked like: a bare
    // media-roots warning, no `error:` line, and no way to tell a timeout from
    // a crash. Print the exit status, both streams and the server log.
    assert!(
        start.status.success(),
        "serve start should succeed.\n  status: {:?}\n  stdout: {}\n  stderr: {}\n  server.log:\n{}",
        start.status,
        String::from_utf8_lossy(&start.stdout),
        String::from_utf8_lossy(&start.stderr),
        std::fs::read_to_string(harness.server_log_path())
            .unwrap_or_else(|e| format!("    <could not read server.log: {e}>"))
    );

    batchalign::server_handshake::ServerHandshake::read(
        harness.state_dir(),
        batchalign::server_handshake::HandshakeSlot::Main,
    )
    .expect("read published handshake")
    .expect("a started server publishes a handshake")
    .bound_port()
    .expect("a started server publishes the port it bound")
    .get()
}

/// A server asked for an ephemeral port publishes the port it actually bound,
/// and that publication is what callers read.
///
/// Asserts the end of the chain that no unit test covers: `port: 0` in the
/// config survives to the bind, the OS picks a port, and the number a caller
/// reads back is the one the server is really listening on. The rationale
/// lives on [`PortRequest`] and in `server_handshake`.
#[test]
fn serve_start_with_ephemeral_port_publishes_the_bound_port() {
    let Some(python_path) = resolve_python() else {
        eprintln!("SKIP: Python 3.12 with batchalign not available");
        return;
    };

    let harness = CliHarness::new();
    // The helper asks for `port: 0` and reads back what was published, which is
    // exactly the property under test, so it is the subject here rather than
    // scaffolding to work around.
    let port = start_daemon_on_ephemeral_port(&harness, &python_path, "auto_daemon: false\n");

    // A real, reachable port: not the 0 that was asked for, and not the 8000
    // that the old silent default would have substituted.
    assert_ne!(port, 8000, "must not fall back to the default port");
    assert!(
        std::net::TcpStream::connect(("127.0.0.1", port)).is_ok(),
        "the published port {port} must be the one the server is listening on"
    );

    let _ = harness
        .cmd()
        .env("BATCHALIGN_PYTHON", &python_path)
        .args(["serve", "stop"])
        .timeout(std::time::Duration::from_secs(10))
        .output();
}

#[test]
fn version() {
    cmd()
        .arg("version")
        .assert()
        .success()
        .stderr(predicate::str::contains(env!("CARGO_PKG_VERSION")));
}

#[test]
fn help_lists_visible_commands() {
    cmd()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("align"))
        .stdout(predicate::str::contains("transcribe"))
        .stdout(predicate::str::contains("compare"))
        .stdout(predicate::str::contains("benchmark"))
        .stdout(predicate::str::contains("morphotag"))
        .stdout(predicate::str::contains("cache"))
        .stdout(predicate::str::contains("logs"))
        .stdout(predicate::str::contains("setup"))
        .stdout(predicate::str::contains("models"))
        .stdout(predicate::str::contains("avqi"))
        .stdout(predicate::str::contains("bench"));
}

#[test]
fn help_per_subcommand() {
    for subcmd in &[
        "align",
        "transcribe",
        "translate",
        "morphotag",
        "coref",
        "utseg",
        "benchmark",
        "opensmile",
        "compare",
        "avqi",
        "setup",
        "serve",
        "jobs",
        "logs",
        "openapi",
        "cache",
        "models",
        "bench",
    ] {
        cmd()
            .args([subcmd, "--help"])
            .assert()
            .success()
            .stdout(predicate::str::is_empty().not());
    }
}

#[test]
fn workflow_command_help_descriptions_match_narrative() {
    for (subcmd, expected) in [
        (
            "align",
            "Align transcripts against corresponding media files",
        ),
        ("transcribe", "Create a transcript from audio files"),
        (
            "morphotag",
            "Perform morphosyntactic analysis on transcripts",
        ),
        (
            "compare",
            "Compare transcripts against gold-standard references",
        ),
        ("benchmark", "Benchmark ASR word accuracy"),
    ] {
        cmd()
            .args([subcmd, "--help"])
            .assert()
            .success()
            .stdout(predicate::str::contains(expected));
    }
}

#[test]
fn verbose_flag() {
    cmd().args(["-vvv", "version"]).assert().success();
}

// ---------------------------------------------------------------------------
// Bench behavior
// ---------------------------------------------------------------------------

#[test]
fn bench_listed_in_help() {
    cmd()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("\n  bench "));
}

#[test]
#[ignore = "integration test: requires running daemon or spawns one with 90s health timeout"]
fn bench_runs_single_iteration() {
    let tmp = tempfile::TempDir::new().unwrap();
    let in_dir = tmp.path().join("in");
    let out_dir = tmp.path().join("out");
    std::fs::create_dir_all(&in_dir).unwrap();
    std::fs::create_dir_all(&out_dir).unwrap();

    cmd()
        .args([
            "bench",
            "align",
            in_dir.to_str().unwrap(),
            out_dir.to_str().unwrap(),
            "--runs",
            "1",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("BENCH_RESULT:"));
}

#[test]
fn setup_non_interactive_writes_config() {
    let harness = CliHarness::new();
    harness
        .cmd()
        .args(["setup", "--non-interactive"])
        .assert()
        .success()
        .stderr(predicate::str::contains("Saved configuration"));

    let cfg = harness.home_dir().join(".batchalign.ini");
    let content = std::fs::read_to_string(cfg).unwrap();
    assert!(content.contains("engine = whisper"));
}

#[test]
fn openapi_writes_output_file() {
    let tmp = tempfile::TempDir::new().unwrap();
    let schema = tmp.path().join("openapi.json");

    cmd()
        .args(["openapi", "--output", schema.to_str().unwrap()])
        .assert()
        .success();

    let content = std::fs::read_to_string(schema).unwrap();
    assert!(content.contains("\"openapi\": \"3.1.0\""));
}

#[test]
fn openapi_check_passes_when_fresh() {
    let tmp = tempfile::TempDir::new().unwrap();
    let schema = tmp.path().join("openapi.json");

    cmd()
        .args(["openapi", "--output", schema.to_str().unwrap()])
        .assert()
        .success();

    cmd()
        .args(["openapi", "--check", "--output", schema.to_str().unwrap()])
        .assert()
        .success()
        .stderr(predicate::str::contains("up to date"));
}

#[test]
fn openapi_check_fails_when_schema_is_stale() {
    let tmp = tempfile::TempDir::new().unwrap();
    let schema = tmp.path().join("openapi.json");
    std::fs::write(&schema, "{}").unwrap();

    cmd()
        .args(["openapi", "--check", "--output", schema.to_str().unwrap()])
        .assert()
        .failure()
        .code(6)
        .stderr(predicate::str::contains("OpenAPI schema is out of date"));
}

// ---------------------------------------------------------------------------
// Models command
// ---------------------------------------------------------------------------

#[test]
fn models_python_module_importable() {
    // The `models` command forwards to `python -m batchalign.models.training.run`.
    // Verify the Python module can at least be found (import check).
    let Some(python) = resolve_python() else {
        eprintln!("SKIP: Python 3.12 with batchalign not available");
        return;
    };

    let output = std::process::Command::new(&python)
        .args(["-c", "import batchalign.models.training.run"])
        .output()
        .unwrap_or_else(|e| panic!("Failed to run {python}: {e}"));

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "`{python} -c 'import batchalign.models.training.run'` failed:\n{stderr}"
    );
}

// ---------------------------------------------------------------------------
// Argument validation
// ---------------------------------------------------------------------------

#[test]
fn unknown_subcommand() {
    cmd().arg("doesnotexist").assert().failure().code(2);
}

#[test]
fn processing_cmd_no_paths() {
    // morphotag with no paths is caught by resolve_inputs (not clap), exits usage(2)
    let harness = CliHarness::new();
    harness
        .cmd()
        .arg("morphotag")
        .assert()
        .failure()
        .code(2)
        .stderr(predicate::str::contains("no input paths"));
}

#[test]
fn processing_cmd_nonexistent_path() {
    let harness = CliHarness::new();
    harness
        .cmd()
        .args(["morphotag", "/nonexistent_path_abc123"])
        .assert()
        .failure()
        .code(2)
        .stderr(predicate::str::contains("does not exist"));
}

#[test]
fn setup_non_interactive_rev_without_key_is_usage_error() {
    cmd()
        .args(["setup", "--non-interactive", "--engine", "rev"])
        .assert()
        .failure()
        .code(2)
        .stderr(predicate::str::contains("requires --rev-key"));
}

#[test]
fn jobs_unreachable_server_is_network_error() {
    cmd()
        .args(["jobs", "--server", "http://127.0.0.1:59999"])
        .assert()
        .failure()
        .code(4);
}

#[test]
fn serve_start_invalid_config_is_config_error() {
    let tmp = tempfile::TempDir::new().unwrap();
    let bad_cfg = tmp.path().join("bad.yaml");
    std::fs::write(&bad_cfg, ":\n  - invalid").unwrap();

    cmd()
        .args([
            "serve",
            "start",
            "--foreground",
            "--config",
            bad_cfg.to_str().unwrap(),
        ])
        .assert()
        .failure()
        .code(3)
        .stderr(predicate::str::contains("failed to parse config"));
}

#[test]
fn serve_start_unknown_config_field_is_config_error() {
    let tmp = tempfile::TempDir::new().unwrap();
    let bad_cfg = tmp.path().join("bad.yaml");
    std::fs::write(&bad_cfg, "port: 9123\nwarmup: false\n").unwrap();

    cmd()
        .args([
            "serve",
            "start",
            "--foreground",
            "--config",
            bad_cfg.to_str().unwrap(),
        ])
        .assert()
        .failure()
        .code(3)
        .stderr(predicate::str::contains("unknown field `warmup`"));
}

// ---------------------------------------------------------------------------

/// Last few KB of a log file, for a failure message.
///
/// Returns why it could not be read rather than an empty string: "the log was
/// absent" and "the log was empty" are different diagnoses, and a silent empty
/// string would let a reader assume the second when it was the first.
fn read_log_tail(path: &std::path::Path) -> String {
    const TAIL_BYTES: usize = 4096;
    match std::fs::read_to_string(path) {
        Ok(text) if text.is_empty() => "(log exists but is empty)".to_string(),
        Ok(text) => {
            let start = text.len().saturating_sub(TAIL_BYTES);
            text[start..].to_string()
        }
        Err(error) => format!("(could not read log: {error})"),
    }
}

// Cache commands (HOME-isolated)
// ---------------------------------------------------------------------------

#[test]
fn cache_stats_no_db() {
    let harness = CliHarness::new();
    harness
        .cmd()
        .args(["cache", "stats"])
        .assert()
        .success()
        .stderr(predicate::str::contains("Entries:"));
}

#[test]
fn cache_clear_no_db() {
    let harness = CliHarness::new();
    harness
        .cmd()
        .args(["cache", "clear", "-y"])
        .assert()
        .success();
}

// ---------------------------------------------------------------------------
// Serve status (no server running)
// ---------------------------------------------------------------------------

#[test]
fn serve_status_unreachable() {
    cmd()
        .args(["serve", "status", "--server", "http://127.0.0.1:59999"])
        .assert()
        .success()
        .stderr(predicate::str::contains("cannot reach"));
}

#[test]
fn serve_status_unknown_config_field_is_config_error() {
    let harness = CliHarness::new();
    harness.write_server_config("port: 9123\nwarmup: false\n");

    harness
        .cmd()
        .args(["serve", "status"])
        .assert()
        .failure()
        .code(3)
        .stderr(predicate::str::contains("unknown field `warmup`"));
}

// ---------------------------------------------------------------------------
// CLI subprocess with real server (P0)
// ---------------------------------------------------------------------------

/// Start a live server and run `batchalign3 morphotag` as a CLI subprocess
/// against it, verifying end-to-end that the binary can talk to a real server,
/// process files, and produce output with %mor and %gra tiers.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cli_morphotag_real_server() {
    use batchalign::worker::InferTask;

    let Some(python_path) = resolve_python() else {
        eprintln!("SKIP: Python 3 with batchalign not available");
        return;
    };

    let tmp = tempfile::TempDir::new().expect("tempdir");
    // Set up input directory with a .cha file
    let in_dir = tmp.path().join("input");
    let out_dir = tmp.path().join("output");
    std::fs::create_dir_all(&in_dir).expect("mkdir input");
    std::fs::create_dir_all(&out_dir).expect("mkdir output");
    std::fs::write(
        in_dir.join("test.cha"),
        "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n@ID:\teng|test|CHI||female|||Target_Child|||\n*CHI:\thello world .\n@End\n",
    ).expect("write input");

    let server = match start_live_server(
        &python_path,
        vec![batchalign_types::paths::ServerPath::new(in_dir.clone())],
    )
    .await
    {
        Ok(server) => server,
        Err(message) => {
            eprintln!("SKIP: {message}");
            return;
        }
    };
    if !server.has_infer_task(InferTask::Morphosyntax) {
        eprintln!("SKIP: live server does not advertise morphosyntax infer support");
        server.shutdown().await;
        return;
    }

    // Run CLI subprocess in a blocking task so it doesn't block the tokio
    // runtime (which is serving the axum server on the same runtime).
    let in_str = in_dir.to_str().unwrap().to_string();
    let out_str = out_dir.to_str().unwrap().to_string();
    let url = server.base_url().to_string();
    let cli_result = tokio::task::spawn_blocking(move || {
        let harness = CliHarness::new();
        let mut command = harness.cmd();
        command
            .args(["morphotag", &in_str, &out_str, "--server", &url])
            .output()
            .expect("spawn CLI")
    })
    .await
    .expect("blocking task");

    server.shutdown().await;

    let stderr = String::from_utf8_lossy(&cli_result.stderr);
    let stdout = String::from_utf8_lossy(&cli_result.stdout);
    eprintln!("CLI stdout: {stdout}");
    eprintln!("CLI stderr: {stderr}");
    assert!(
        cli_result.status.success(),
        "CLI should succeed. stderr: {stderr}"
    );

    // The CLI should succeed and report "All done!" or "written".
    // With test-echo workers, the morphosyntax pipeline may not produce
    // valid output (test-echo echoes raw IPC, not NLP results), so we
    // only verify the CLI completes without error.
    assert!(
        stderr.contains("All done") || stderr.contains("written"),
        "CLI should report completion. stderr: {stderr}"
    );

    let output_chat =
        std::fs::read_to_string(out_dir.join("test.cha")).expect("read morphotag output");
    assert!(
        output_chat.contains("%mor:"),
        "morphotag output should contain a %mor tier. output:\n{output_chat}"
    );
    assert!(
        output_chat.contains("%gra:"),
        "morphotag output should contain a %gra tier. output:\n{output_chat}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cli_align_real_server_live_fa_succeeds() {
    use batchalign::worker::InferTask;

    let Some(python_path) = resolve_python() else {
        eprintln!("SKIP: Python 3.12 with batchalign not available");
        return;
    };

    let repo = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crate dir parent")
        .parent()
        .expect("repo root");
    let source_audio = repo.join("batchalign/tests/support/test.mp3");
    let source_chat = repo.join("batchalign/tests/formats/chat/support/test.cha");
    if !source_audio.is_file() || !source_chat.is_file() {
        eprintln!("SKIP: committed align fixture pair is unavailable");
        return;
    }

    let tmp = tempfile::TempDir::new().expect("tempdir");
    let in_dir = tmp.path().join("input");
    let out_dir = tmp.path().join("output");
    std::fs::create_dir_all(&in_dir).expect("mkdir input");
    std::fs::create_dir_all(&out_dir).expect("mkdir output");
    std::fs::copy(&source_audio, in_dir.join("test.mp3")).expect("copy audio");

    let stripped_chat = std::fs::read_to_string(&source_chat)
        .expect("read source chat")
        .lines()
        .filter(|line| {
            !(line.starts_with("%mor:") || line.starts_with("%gra:") || line.starts_with("%wor:"))
        })
        .collect::<Vec<_>>()
        .join("\n")
        + "\n";
    std::fs::write(in_dir.join("test.cha"), stripped_chat).expect("write stripped chat");

    let server = match start_live_server(
        &python_path,
        vec![batchalign_types::paths::ServerPath::new(in_dir.clone())],
    )
    .await
    {
        Ok(server) => server,
        Err(message) => {
            eprintln!("SKIP: {message}");
            return;
        }
    };
    if !server.has_infer_task(InferTask::Fa) {
        eprintln!("SKIP: live server does not advertise FA infer support");
        server.shutdown().await;
        return;
    }

    let in_str = in_dir.to_str().unwrap().to_string();
    let out_str = out_dir.to_str().unwrap().to_string();
    let url = server.base_url().to_string();
    let cli_result = tokio::task::spawn_blocking(move || {
        let harness = CliHarness::new();
        let mut command = harness.cmd();
        command
            .args([
                "align",
                &in_str,
                "-o",
                &out_str,
                "--server",
                &url,
                "--whisper-fa",
                "--no-utr",
            ])
            .timeout(std::time::Duration::from_secs(300))
            .output()
            .expect("spawn CLI")
    })
    .await
    .expect("blocking task");

    server.shutdown().await;

    let stderr = String::from_utf8_lossy(&cli_result.stderr);
    let stdout = String::from_utf8_lossy(&cli_result.stdout);
    eprintln!("CLI stdout: {stdout}");
    eprintln!("CLI stderr: {stderr}");
    assert!(
        cli_result.status.success(),
        "CLI should succeed. stdout: {stdout}\nstderr: {stderr}"
    );
    assert!(
        stderr.contains("All done! 1 file(s) written"),
        "CLI should report a completed write. stderr: {stderr}"
    );

    let output_chat =
        std::fs::read_to_string(out_dir.join("test.cha")).expect("read aligned output");
    assert!(
        output_chat.contains("%wor:"),
        "aligned output should contain a %wor tier. output:\n{output_chat}"
    );
}

#[test]
fn cli_transcribe_explicit_server_never_falls_back_to_local_daemon() {
    let Some(python_path) = resolve_python() else {
        eprintln!("SKIP: Python 3.12 with batchalign not available");
        return;
    };

    let harness = CliHarness::new();
    let in_dir = harness.home_dir().join("input");
    let out_dir = harness.home_dir().join("output");
    std::fs::create_dir_all(&in_dir).expect("mkdir input");
    std::fs::create_dir_all(&out_dir).expect("mkdir output");
    write_silent_wav(&in_dir.join("sample.wav"));

    let port = start_daemon_on_ephemeral_port(&harness, &python_path, "auto_daemon: true\n");

    let cli_result = harness
        .cmd()
        .env("BATCHALIGN_PYTHON", &python_path)
        .args([
            "transcribe",
            in_dir.to_str().unwrap(),
            "-o",
            out_dir.to_str().unwrap(),
            "--lang",
            "eng",
            "--server",
            "http://127.0.0.1:59999",
        ])
        .timeout(std::time::Duration::from_secs(60))
        .output()
        .expect("spawn CLI");

    let _ = harness
        .cmd()
        .env("BATCHALIGN_PYTHON", &python_path)
        .args(["serve", "stop"])
        .timeout(std::time::Duration::from_secs(10))
        .output();

    let stderr = String::from_utf8_lossy(&cli_result.stderr);
    let stdout = String::from_utf8_lossy(&cli_result.stdout);
    eprintln!("CLI stdout: {stdout}");
    eprintln!("CLI stderr: {stderr}");
    assert!(
        !cli_result.status.success(),
        "an unreachable explicit server must fail. stdout: {stdout}\nstderr: {stderr}"
    );
    assert!(
        !stderr.contains(&format!(
            "Submitting to local daemon at http://127.0.0.1:{port}"
        )),
        "the healthy local daemon must not override --server. stderr: {stderr}"
    );
    assert!(
        !out_dir.join("sample.cha").is_file(),
        "no transcript may be attributed to an explicit server that was never reached"
    );
}

/// Audio commands cannot target a non-loopback server until the CLI has a
/// real media-body upload transport. Refusal must happen before HTTP so a
/// same-named recording on that server cannot be mistaken for this input.
#[test]
fn cli_transcribe_explicit_remote_server_refuses_unshippable_audio() {
    let harness = CliHarness::new();
    let input = harness.home_dir().join("sample.wav");
    let output = harness.home_dir().join("output");
    write_silent_wav(&input);

    let result = harness
        .cmd()
        .args([
            "transcribe",
            input.to_str().unwrap(),
            "-o",
            output.to_str().unwrap(),
            "--server",
            "https://worker.example.org",
        ])
        .timeout(std::time::Duration::from_secs(3))
        .output()
        .expect("spawn CLI");

    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(!result.status.success(), "remote audio must be refused");
    assert!(
        stderr.contains("cannot send local audio to non-loopback server"),
        "refusal must name the unsupported transport before HTTP. stderr: {stderr}"
    );
}

#[test]
fn cli_align_explicit_loopback_uses_shared_paths_mode() {
    let Some(python_path) = resolve_python() else {
        eprintln!("SKIP: Python 3.12 with batchalign not available");
        return;
    };

    let harness = CliHarness::new();
    let media_dir = harness.home_dir().join("media");
    let in_dir = harness.home_dir().join("input");
    let out_dir = harness.home_dir().join("output");
    std::fs::create_dir_all(&media_dir).expect("mkdir media");
    std::fs::create_dir_all(&in_dir).expect("mkdir input");
    std::fs::create_dir_all(&out_dir).expect("mkdir output");
    write_silent_wav(&media_dir.join("sample.wav"));
    std::fs::write(
        in_dir.join("sample.cha"),
        "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tPAR Participant\n@ID:\teng|test|PAR|||||Participant|||\n@Media:\tsample, audio\n*PAR:\thello world .\n@End\n",
    )
    .expect("write input");

    let port = start_daemon_on_ephemeral_port(
        &harness,
        &python_path,
        &format!(
            "auto_daemon: false\nmedia_roots:\n  - {}\n",
            media_dir.display()
        ),
    );

    let cli_result = harness
        .cmd()
        .env("BATCHALIGN_PYTHON", &python_path)
        .args([
            "align",
            in_dir.to_str().unwrap(),
            "-o",
            out_dir.to_str().unwrap(),
            "--server",
            &format!("http://127.0.0.1:{port}"),
        ])
        .timeout(std::time::Duration::from_secs(60))
        .output()
        .expect("spawn CLI");

    let _ = harness
        .cmd()
        .env("BATCHALIGN_PYTHON", &python_path)
        .args(["serve", "stop"])
        .timeout(std::time::Duration::from_secs(10))
        .output();

    let stderr = String::from_utf8_lossy(&cli_result.stderr);
    let stdout = String::from_utf8_lossy(&cli_result.stdout);
    eprintln!("CLI stdout: {stdout}");
    eprintln!("CLI stderr: {stderr}");
    assert!(
        cli_result.status.success(),
        "align should succeed against the explicit loopback test server. stdout: {stdout}\nstderr: {stderr}"
    );
    assert!(
        stderr.contains(&format!(
            "Submitting shared-filesystem job to http://127.0.0.1:{port}"
        )),
        "an explicit loopback server should use shared paths. stderr: {stderr}"
    );
    assert!(
        !stderr.contains("ignoring --server and using local daemon"),
        "align should stay on the explicit server path. stderr: {stderr}"
    );
    assert!(
        out_dir.join("sample.cha").is_file(),
        "align should write the echoed output file via the explicit server path"
    );
    let output_chat =
        std::fs::read_to_string(out_dir.join("sample.cha")).expect("read aligned output");
    assert!(
        output_chat.contains("*PAR:\thello world ."),
        "align shared-path output should preserve the source CHAT content. output:\n{output_chat}"
    );
}

#[test]
fn cli_transcribe_in_place_mp4_succeeds_via_local_daemon() {
    let Some(python_path) = resolve_python() else {
        eprintln!("SKIP: Python 3.12 with batchalign not available");
        return;
    };
    if MediaTool::Ffmpeg.banner().is_none() {
        eprintln!("SKIP: ffmpeg not installed");
        return;
    }

    let harness = CliHarness::new();
    let media_file = harness.home_dir().join("input").join("clip.mp4");
    std::fs::create_dir_all(media_file.parent().expect("media parent")).expect("mkdir input");
    write_silent_mp4(&media_file);

    let _port = start_daemon_on_ephemeral_port(&harness, &python_path, "auto_daemon: true\n");

    let cli_result = harness
        .cmd()
        .env("BATCHALIGN_PYTHON", &python_path)
        .args([
            "transcribe",
            "--in-place",
            media_file.to_str().unwrap(),
            "--lang",
            "eng",
        ])
        .timeout(std::time::Duration::from_secs(60))
        .output()
        .expect("spawn CLI");

    let _ = harness
        .cmd()
        .env("BATCHALIGN_PYTHON", &python_path)
        .args(["serve", "stop"])
        .timeout(std::time::Duration::from_secs(10))
        .output();

    let stderr = String::from_utf8_lossy(&cli_result.stderr);
    let stdout = String::from_utf8_lossy(&cli_result.stdout);
    eprintln!("CLI stdout: {stdout}");
    eprintln!("CLI stderr: {stderr}");
    assert!(
        cli_result.status.success(),
        "CLI should succeed. stdout: {stdout}\nstderr: {stderr}"
    );
    assert!(
        stderr.contains("All done! 1 file(s) written"),
        "CLI should report a completed write. stderr: {stderr}"
    );
    assert!(
        media_file.with_extension("cha").is_file(),
        "in-place mp4 transcription should write a sibling CHAT file"
    );
}

#[test]
fn cli_transcribe_in_place_mp4_populates_injected_media_cache_live() {
    let Some(python_path) = resolve_python() else {
        eprintln!("SKIP: Python 3.12 with batchalign not available");
        return;
    };
    if MediaTool::Ffmpeg.banner().is_none() {
        eprintln!("SKIP: ffmpeg not installed");
        return;
    }

    let repo = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crate dir parent")
        .parent()
        .expect("repo root");
    let source_audio = repo.join("batchalign/tests/support/test.mp3");
    if !source_audio.is_file() {
        eprintln!("SKIP: committed speech fixture is unavailable");
        return;
    }

    let harness = CliHarness::new();
    let media_file = harness.home_dir().join("input").join("clip.mp4");
    let cache_dir = harness.home_dir().join("media-cache");
    std::fs::create_dir_all(media_file.parent().expect("media parent")).expect("mkdir input");
    std::fs::create_dir_all(&cache_dir).expect("mkdir cache");
    transcode_audio_to_mp4(&source_audio, &media_file);

    // Ephemeral: the auto-daemon binds whatever the OS gives it and publishes
    // the result, so no port number is reserved or handed over and there is no
    // window for another process to take one.
    harness.write_server_config("host: 127.0.0.1\nport: 0\nauto_daemon: true\n");

    let cli_result = harness
        .cmd()
        .env("BATCHALIGN_PYTHON", &python_path)
        .env("BATCHALIGN_MEDIA_CACHE_DIR", &cache_dir)
        .args([
            "transcribe",
            "--in-place",
            media_file.to_str().unwrap(),
            "--lang",
            "eng",
            "--whisper",
        ])
        .timeout(std::time::Duration::from_secs(300))
        .output()
        .expect("spawn CLI");

    // Read the port the daemon published, before stopping it. This is the same
    // number the CLI printed, read from the daemon's own record rather than
    // predicted by the test.
    let port = published_daemon_port(&harness);

    let _ = harness
        .cmd()
        .env("BATCHALIGN_PYTHON", &python_path)
        .args(["serve", "stop"])
        .timeout(std::time::Duration::from_secs(10))
        .output();

    let stderr = String::from_utf8_lossy(&cli_result.stderr);
    let stdout = String::from_utf8_lossy(&cli_result.stdout);
    eprintln!("CLI stdout: {stdout}");
    eprintln!("CLI stderr: {stderr}");
    assert!(
        cli_result.status.success(),
        "CLI should succeed. stdout: {stdout}\nstderr: {stderr}"
    );
    assert!(
        stderr.contains(&format!(
            "Submitting to local daemon at http://127.0.0.1:{port}"
        )),
        // The CLI's own advice on this failure is "check daemon.log", so read it
        // here rather than telling a future reader to go and find a temp
        // directory that the harness deletes on drop. Without this the panic
        // says only "could not start local daemon", which is the symptom the
        // CLI already printed and names no cause. Observed 2026-07-31 under
        // full-suite load; the same test passes alone in ~125s.
        "CLI should use the isolated local daemon. stderr: {stderr}\n\
         --- daemon.log ({daemon_log}) ---\n{daemon_log_tail}",
        daemon_log = harness.home_dir().join(".batchalign3/daemon.log").display(),
        daemon_log_tail = read_log_tail(&harness.home_dir().join(".batchalign3/daemon.log")),
    );
    assert!(
        stderr.contains("All done! 1 file(s) written"),
        "CLI should report a completed write. stderr: {stderr}"
    );

    let output_chat = std::fs::read_to_string(media_file.with_extension("cha"))
        .expect("read in-place transcribe output");
    let cached_wavs = std::fs::read_dir(&cache_dir)
        .expect("read cache dir")
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().is_some_and(|ext| ext == "wav"))
        .collect::<Vec<_>>();
    assert!(
        !cached_wavs.is_empty(),
        "real mp4 transcribe should populate the injected media cache. stderr: {stderr}"
    );
    assert!(
        output_chat.contains("@Media:\tclip, audio"),
        "output should preserve the original media basename even when a cached WAV is used internally. output:\n{output_chat}"
    );
}

/// An explicit loopback compare server should keep the named producer while
/// using the shared filesystem.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cli_compare_explicit_loopback_uses_shared_paths_mode() {
    let Some(_python_path) = resolve_python() else {
        eprintln!("SKIP: Python 3 with batchalign not available");
        return;
    };

    let Some(session) = acquire_test_server_session().await else {
        return;
    };
    let server_url = session.base_url().to_owned();
    let harness = CliHarness::new();
    let in_dir = harness.home_dir().join("input");
    let out_dir = harness.home_dir().join("output");
    std::fs::create_dir_all(&in_dir).expect("mkdir input");
    std::fs::create_dir_all(&out_dir).expect("mkdir output");
    std::fs::write(in_dir.join("test.cha"), MINIMAL_CHAT).expect("write input");

    let home = harness.home_dir().to_path_buf();
    let in_str = in_dir.to_str().unwrap().to_string();
    let out_str = out_dir.to_str().unwrap().to_string();
    let url = server_url.clone();
    let cli_result = tokio::task::spawn_blocking(move || {
        let mut command = cmd();
        command.env("HOME", &home);
        command
            .args(["compare", &in_str, &out_str, "--server", &url])
            .output()
            .expect("spawn CLI")
    })
    .await
    .expect("blocking task");

    let stderr = String::from_utf8_lossy(&cli_result.stderr);
    let stdout = String::from_utf8_lossy(&cli_result.stdout);
    eprintln!("CLI stdout: {stdout}");
    eprintln!("CLI stderr: {stderr}");
    assert!(
        cli_result.status.success(),
        "compare should succeed against the explicit loopback test server. stdout: {stdout}\nstderr: {stderr}"
    );
    assert!(
        stderr.contains(&format!("Submitting shared-filesystem job to {server_url}")),
        "compare should use shared paths on the explicit loopback server. stderr: {stderr}"
    );
    assert!(
        !stderr.contains("Submitting to local daemon"),
        "compare explicit server path should not route through the local daemon. stderr: {stderr}"
    );
    assert!(
        out_dir.join("test.cha").is_file(),
        "compare should write the echoed output file via the explicit server path"
    );
}

#[test]
fn cli_compare_failed_auto_daemon_job_returns_server_exit_code() {
    let Some(python_path) = resolve_python() else {
        eprintln!("SKIP: Python 3 with batchalign not available");
        return;
    };

    let harness = CliHarness::new();
    let in_dir = harness.home_dir().join("input");
    let out_dir = harness.home_dir().join("output");
    std::fs::create_dir_all(&in_dir).expect("mkdir input");
    std::fs::create_dir_all(&out_dir).expect("mkdir output");
    std::fs::write(in_dir.join("test.cha"), MINIMAL_CHAT).expect("write input");

    // Ephemeral: the auto-daemon binds whatever the OS gives it and publishes
    // the result, so no port number is reserved or handed over and there is no
    // window for another process to take one.
    harness.write_server_config("host: 127.0.0.1\nport: 0\nauto_daemon: true\n");

    let cli_result = harness
        .cmd()
        .env("BATCHALIGN_PYTHON", &python_path)
        .args([
            "compare",
            in_dir.to_str().unwrap(),
            out_dir.to_str().unwrap(),
        ])
        // 300s, not 120s. This bound is a HARNESS SAFETY NET, not the property
        // under test: the assertion below is about the exit CODE a failed
        // auto-daemon job maps to, and no timeout value makes that assertion
        // weaker. At 120s it fired under parallel load on 2026-07-28 while the
        // test passed in 35.7s alone, and a harness kill surfaces as
        // `status.code() == None`, which reads as a baffling `None != Some(5)`
        // rather than "we ran out of time". Raising the net costs nothing;
        // the explicit None check below makes the distinction legible.
        .timeout(std::time::Duration::from_secs(300))
        .output()
        .expect("spawn CLI");

    let _ = harness
        .cmd()
        .env("BATCHALIGN_PYTHON", &python_path)
        .args(["serve", "stop"])
        .timeout(std::time::Duration::from_secs(10))
        .output();

    let stderr = String::from_utf8_lossy(&cli_result.stderr);
    let stdout = String::from_utf8_lossy(&cli_result.stdout);
    eprintln!("CLI stdout: {stdout}");
    eprintln!("CLI stderr: {stderr}");
    // Distinguish "killed by the harness timeout" from "wrong exit code".
    // Without this, a slow machine reports the same failure text as a genuine
    // exit-code regression, which is how this test spent time being diagnosed
    // as a cancellation bug it never was.
    assert!(
        cli_result.status.code().is_some(),
        "CLI was terminated by a signal rather than exiting, almost certainly \
         the harness timeout on a loaded machine. This is NOT an exit-code \
         regression. stdout: {stdout}\nstderr: {stderr}"
    );
    assert_eq!(
        cli_result.status.code(),
        Some(5),
        "failed auto-daemon job should map to exit code 5. stdout: {stdout}\nstderr: {stderr}"
    );
    assert!(
        stderr.contains("status failed") || stderr.contains("gold") || stderr.contains("error:"),
        "CLI stderr should describe the failed job. stderr: {stderr}"
    );
}

/// Helpful-error contract: a bad input path must be NAMED in the error.
///
/// Regression guard for the 2026-07-10 field failure: `align` with a
/// path that failed client-side resolution printed a bare
/// "No such file or directory (os error 2)" with no path and no hint
/// (std's `canonicalize` errors are pathless, and `CliError::Io`
/// displayed them raw), leaving the operator to bisect by hand.
#[test]
fn missing_input_path_error_names_the_path() {
    let harness = CliHarness::new();
    let tmp = tempfile::TempDir::new().unwrap();

    let result = harness
        .cmd()
        .current_dir(tmp.path())
        .args(["align", "no-such-file.cha"])
        .timeout(std::time::Duration::from_secs(30))
        .output()
        .unwrap();

    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(
        !result.status.success(),
        "align on a missing file must fail. stderr: {stderr}"
    );
    assert!(
        stderr.contains("no-such-file.cha"),
        "the error must NAME the offending path, not print a bare io error. stderr: {stderr}"
    );
}

/// The 2026-07-10 field failure, exactly: an EXISTING bare relative
/// filename (no `./`, cwd elsewhere) in in-place output mode. The
/// output-parent derivation yields the empty path, and
/// `create_dir_all("")` fails ENOENT client-side, printed pathless.
/// Contract: this must either work (treat empty parent as `.`) or
/// fail naming the path; a bare "No such file or directory" is a bug.
#[test]
fn bare_relative_input_does_not_fail_pathless() {
    let harness = CliHarness::new();
    let tmp = tempfile::TempDir::new().unwrap();
    std::fs::write(
        tmp.path().join("session.cha"),
        "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n@ID:\teng|corpus|CHI|||||Target_Child|||\n*CHI:\thello .\n@End\n",
    )
    .unwrap();

    let result = harness
        .cmd()
        .current_dir(tmp.path())
        .args(["align", "session.cha"])
        .timeout(std::time::Duration::from_secs(60))
        .output()
        .unwrap();

    let stderr = String::from_utf8_lossy(&result.stderr);
    if !result.status.success() {
        assert!(
            stderr.contains("session.cha") || stderr.contains("media") || stderr.contains("daemon"),
            "failure must name the path or a real cause, never a bare io error. stderr: {stderr}"
        );
        assert!(
            !stderr.contains("(os error 2)") || stderr.contains("session.cha"),
            "pathless ENOENT reached the user. stderr: {stderr}"
        );
    }
}

/// Contract: the `diarize` subcommand exists and describes its job.
///
/// V1 scope (workflow-level wiring): media in, speaker-turns JSON out,
/// consumable by `chatter rediarize --turns`. The in-crate composition
/// (applying the turns to a transcript in the same command) waits on
/// the next chatter release carrying the rediarize transform.
#[test]
fn diarize_help_describes_speaker_turns() {
    let harness = CliHarness::new();
    let result = harness
        .cmd()
        .args(["diarize", "--help"])
        .timeout(std::time::Duration::from_secs(30))
        .output()
        .unwrap();

    assert!(
        result.status.success(),
        "diarize --help must exist and exit 0. stderr: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    let stdout = String::from_utf8_lossy(&result.stdout).to_lowercase();
    assert!(
        stdout.contains("diarization") && stdout.contains("turns"),
        "help must say what the command produces. stdout: {stdout}"
    );
    assert!(
        stdout.contains("--speaker-engine")
            && stdout.contains("pyannote-ai")
            && stdout.contains("pyannote"),
        "the binary help must expose local and paid standalone backends. stdout: {stdout}"
    );
}

/// Contract: a missing media input fails with an error naming the path,
/// never a pathless io error (same family as the align contract above).
#[test]
fn diarize_missing_media_error_names_the_path() {
    let harness = CliHarness::new();
    let result = harness
        .cmd()
        .args(["diarize", "/nonexistent/dir/absent.wav"])
        .timeout(std::time::Duration::from_secs(60))
        .output()
        .unwrap();

    assert!(
        !result.status.success(),
        "diarizing a missing file must fail"
    );
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(
        stderr.contains("absent.wav"),
        "failure must name the missing media path. stderr: {stderr}"
    );
    assert!(
        !stderr.contains("unrecognized subcommand"),
        "diarize must be a real subcommand. stderr: {stderr}"
    );
}

/// Contract: the `speaker-identify` subcommand exists and describes what it
/// produces, including that it does not rewrite the transcript.
#[test]
fn speaker_identify_help_describes_enrollment_and_evidence() {
    let harness = CliHarness::new();
    let result = harness
        .cmd()
        .args(["speaker-identify", "--help"])
        .timeout(std::time::Duration::from_secs(30))
        .output()
        .unwrap();

    assert!(
        result.status.success(),
        "speaker-identify --help must exist and exit 0. stderr: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    let stdout = String::from_utf8_lossy(&result.stdout).to_lowercase();
    assert!(
        stdout.contains("--enroll") && stdout.contains("--threshold"),
        "help must expose the two arguments the command cannot run without. stdout: {stdout}"
    );
    assert!(
        stdout.contains("<start_ms>-<end_ms>:<label>"),
        "help must state the enrollment form. stdout: {stdout}"
    );
}

/// Contract: the threshold is REQUIRED at the real CLI boundary.
///
/// The unit test pins clap's `required = true`; this pins that the shipped
/// binary actually refuses, which is the thing an operator meets. A default
/// slipping back in would make every run produce confident verdicts under a
/// number nobody chose.
#[test]
fn speaker_identify_refuses_to_run_without_a_threshold() {
    let harness = CliHarness::new();
    let result = harness
        .cmd()
        .args(["speaker-identify", "corpus/", "--enroll", "1500-9000:INV"])
        .timeout(std::time::Duration::from_secs(30))
        .output()
        .unwrap();

    assert!(!result.status.success(), "a missing threshold must fail");
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(
        stderr.contains("--threshold"),
        "the refusal must name the missing argument. stderr: {stderr}"
    );
    assert!(
        !stderr.contains("unrecognized subcommand"),
        "speaker-identify must be a real subcommand. stderr: {stderr}"
    );
}

/// Contract: two enrollments claiming the same audio for different speakers
/// are refused by the shipped binary, with the overlap named.
///
/// This is the cross-argument check `cli::run_command` performs before typed
/// options are built. No per-value parser can see it, so only an end-to-end
/// invocation proves the guard is wired.
#[test]
fn speaker_identify_refuses_overlapping_enrollments() {
    let harness = CliHarness::new();
    let corpus = harness.home_dir().join("corpus");
    std::fs::create_dir(&corpus).expect("create input corpus");
    let result = harness
        .cmd()
        .arg("speaker-identify")
        .arg(corpus)
        .args([
            "--enroll",
            "0-5000:INV",
            "--enroll",
            "4000-9000:CHI",
            "--threshold",
            "0.5",
        ])
        .timeout(std::time::Duration::from_secs(60))
        .output()
        .unwrap();

    assert!(
        !result.status.success(),
        "overlapping enrollments must fail"
    );
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(
        stderr.contains("overlap"),
        "the refusal must say what is wrong. stderr: {stderr}"
    );
}
