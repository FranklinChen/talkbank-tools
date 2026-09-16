//! Dispatch router: routes processing commands to direct or server hosts.
//!
//! Mirrors `dispatch.py` + `dispatch_server.py`.
//!
//! Explicit `--server` always selects that HTTP server. Without an explicit
//! target, local processing commands can auto-spawn a daemon or execute inline
//! through the shared direct host.

mod helpers;
mod paths;
mod single;

use std::time::Duration;

use crate::config::{RuntimeLayout, load_validated_config_from_layout};
use crate::host_facts::EffectiveConfig;
use crate::host_memory::HostMemoryRuntimeConfig;
use crate::host_policy::HostExecutionPolicy;
use crate::options::CommandOptions;
use crate::worker::handle::WorkerRuntimeConfig;
use crate::worker::pool::PoolConfig;
use crate::{DirectHost, RegistryDiscovery, ReleasedCommand, prepare_workers};
use crate::{api::JobInfo, api::JobStatus, config::ServerConfig};

use crate::cli::client::{self, BatchalignClient};
use crate::cli::daemon;
use crate::cli::error::CliError;
use crate::cli::progress::BatchProgress;
use crate::cli::python::resolve_python_executable;

use crate::cli::args::InputKind;
use helpers::{DirectProgressTracker, file_error_details, finish_terminal_job};
use paths::prepare_paths_submission;
use single::{ServerTarget, dispatch_single_server};

// ---------------------------------------------------------------------------
// Top-level dispatch router
// ---------------------------------------------------------------------------

/// Named dispatch request for one CLI processing invocation.
#[derive(Debug)]
pub struct DispatchRequest<'a> {
    /// Canonical processing command name.
    pub command: ReleasedCommand,
    /// Primary language for the command.
    pub lang: &'a str,
    /// Requested number of speakers.
    pub num_speakers: u32,
    /// File extensions to discover.
    pub input_kind: InputKind,
    /// Explicit remote server URL, if any.
    pub server_arg: Option<&'a str>,
    /// Input paths supplied on the CLI.
    pub inputs: &'a [std::path::PathBuf],
    /// Optional output directory.
    pub out_dir: Option<&'a std::path::Path>,
    /// Typed command options for submission.
    pub options: Option<CommandOptions>,
    /// Optional TalkBank bank name.
    pub bank: Option<&'a str>,
    /// Optional bank subdirectory.
    pub subdir: Option<&'a str>,
    /// Optional lexicon path.
    pub lexicon: Option<&'a str>,
    /// Whether to use the TUI.
    pub use_tui: bool,
    /// Whether to auto-open the dashboard.
    pub open_dashboard: bool,
    /// Whether to force CPU execution for local worker processes.
    pub force_cpu: bool,
    /// Explicit Apple-GPU opt-in (`--allow-mps`), threaded to workers.
    pub allow_mps: bool,
    /// Skip auto-detection of a local server (force direct mode).
    pub no_server: bool,
    /// Optional before-path input for incremental workflows.
    pub before: Option<&'a std::path::Path>,
    /// Optional explicit worker count.
    pub workers: Option<usize>,
    /// Optional daemon startup timeout.
    pub timeout: Option<u64>,
    /// Sequential mode: one worker, no memory gate, no server.
    pub sequential: bool,
    /// Override auto-detected memory tier (small, medium, large, fleet).
    pub memory_tier: Option<crate::types::runtime::MemoryTierKind>,
}

/// Route a processing command to the appropriate execution host.
///
/// # Why `layout` is a parameter and not ambient
///
/// The runtime layout selects the state directory, and therefore which
/// `server.yaml` this dispatch obeys. Resolving it inside from
/// `BATCHALIGN_STATE_DIR` made it ambient PROCESS state, so the only way for a
/// caller to steer it was to mutate its own environment. That is the same
/// defect already cured on the worker spawn path by
/// `WorkerRuntimeConfig::state_dir`,
/// where three tests mutating process env raced each other under cargo's
/// parallel threads and produced off-by-one daemon counts that read like
/// registry bugs.
///
/// Taking it here makes the choice explicit at each call site: the binary
/// passes `RuntimeLayout::from_env()` and says so, and a test passes
/// `RuntimeLayout::from_state_dir(dir)` without touching shared state.
///
/// This is the main entry point for all CLI processing commands. It resolves
/// where to send work using the following priority chain:
///
/// 1. **Explicit `--server URL`** -- authoritative single-server dispatch via
///    HTTP. Loopback targets use shared-filesystem transport without changing
///    the selected producer.
/// 2. **Local daemon** -- when `auto_daemon` is enabled, the CLI reuses or
///    starts a loopback daemon and routes eligible commands through it.
/// 3. **Already-running local server** -- if a loopback server is listening on
///    the configured port, reuse it without spawning a daemon.
/// 4. **Direct local execution** -- local filesystem processing goes through
///    the shared direct host with no daemon/queue layer.
///
/// # Parameters
///
/// Takes one [`DispatchRequest`] describing the command profile, input/output
/// paths, typed options, and UI/runtime toggles for this CLI invocation.
///
/// # Errors
///
/// Returns [`CliError`] on I/O failures, HTTP errors, job failures, or direct
/// execution failures.
pub async fn dispatch(
    request: DispatchRequest<'_>,
    layout: &RuntimeLayout,
) -> Result<(), CliError> {
    let DispatchRequest {
        command,
        lang,
        num_speakers,
        input_kind,
        server_arg,
        inputs,
        out_dir,
        options,
        bank,
        subdir,
        lexicon,
        use_tui,
        open_dashboard,
        force_cpu,
        allow_mps,
        no_server,
        before,
        workers,
        timeout,
        sequential,
        memory_tier,
    } = request;

    let no_server = no_server || sequential;
    let workers = if sequential { Some(1) } else { workers };
    if sequential && server_arg.is_some() {
        return Err(CliError::InvalidArgument(
            "--sequential and --server are mutually exclusive".into(),
        ));
    }

    if bank.is_some() || subdir.is_some() {
        eprintln!(
            "error: --bank/--subdir remote media selection is no longer supported.\n\
             Pass filesystem paths that are visible on the execution host instead."
        );
        return Ok(());
    }

    // 1. An explicit --server always identifies the execution host. Loopback
    // targets can safely send filesystem paths; other targets use content and
    // media-reference submission. Falling through to a healthy local daemon
    // after the operator named another server makes the producer identity
    // false and can send paid work to the wrong process.
    if let Some(server) = server_arg {
        let client = BatchalignClient::new()?;
        let urls = client::parse_servers(server);
        if urls.is_empty() {
            eprintln!("error: no server URL provided");
            return Ok(());
        }

        if urls.len() == 1 {
            let target = ServerTarget::parse_explicit(&urls[0], command)?;
            return dispatch_single_server(
                &client,
                &target,
                command,
                lang,
                num_speakers,
                input_kind,
                inputs,
                out_dir,
                options.as_ref(),
                lexicon,
                before,
                use_tui,
                open_dashboard,
            )
            .await;
        }

        eprintln!(
            "error: multi-server dispatch (--server URL1,URL2) is not available in this version.\n\
             Use --server with a single URL instead."
        );
        return Ok(());
    }

    let (mut cfg, warnings) = load_validated_config_from_layout(layout, None)?;
    for warning in warnings {
        eprintln!("warning: {warning}");
    }

    // Apply --memory-tier CLI override (for testing constrained-memory behavior
    // on large machines). Overrides auto-detection from system RAM.
    if let Some(tier) = memory_tier {
        cfg.memory_tier = Some(tier);
    }

    // 2. Auto-daemon routing applies only when no server was explicitly named.
    let local_daemon_url = if !no_server && cfg.auto_daemon {
        daemon::ensure_daemon(force_cpu, allow_mps, workers, timeout).await?
    } else {
        None
    };

    if let Some(local_daemon_url) = local_daemon_url.as_deref() {
        eprintln!("Submitting to local daemon at {local_daemon_url}\n");
        let client = BatchalignClient::new()?;
        let target = ServerTarget::parse_shared_filesystem(local_daemon_url)?;
        return dispatch_single_server(
            &client,
            &target,
            command,
            lang,
            num_speakers,
            input_kind,
            inputs,
            out_dir,
            options.as_ref(),
            lexicon,
            before,
            use_tui,
            open_dashboard,
        )
        .await;
    }

    // 3. Auto-detect a loopback server that was started outside the daemon
    // state-file flow (for example by launchd or a foreground serve command).
    // Prefer the port the server published over the one the config asked for.
    // Computed INSIDE the guard: `--no-server` and `--sequential` never contact
    // a server, so resolving one was pure waste on exactly the paths that opted
    // out. `cfg` is already loaded, so this costs one small file read.
    if !no_server
        && let Some(local_url) = daemon::local_server_url(layout, cfg.port)
        && let Some(health) = probe_local_server(&local_url).await
    {
        eprintln!("Using local server at {local_url} ({health})\n");
        let client = BatchalignClient::new()?;
        let target = ServerTarget::parse_shared_filesystem(&local_url)?;
        return dispatch_single_server(
            &client,
            &target,
            command,
            lang,
            num_speakers,
            input_kind,
            inputs,
            out_dir,
            options.as_ref(),
            lexicon,
            before,
            use_tui,
            open_dashboard,
        )
        .await;
    }

    // 4. If we reached here without --no-server or --sequential the user did
    //    not explicitly request offline execution. Exit gracefully and create
    //    the output directory so tools that check for its existence still
    //    see a consistent post-run filesystem state.
    if !no_server {
        if let Some(dir) = out_dir {
            let _ = std::fs::create_dir_all(dir);
        }
        eprintln!("no server available");
        return Ok(());
    }

    // Direct local execution (--no-server / --sequential path)
    dispatch_direct_mode(
        cfg,
        layout.clone(),
        command,
        lang,
        num_speakers,
        input_kind,
        inputs,
        out_dir,
        options.as_ref(),
        lexicon,
        before,
        force_cpu,
        allow_mps,
        workers,
        timeout,
        sequential,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn dispatch_direct_mode(
    mut cfg: ServerConfig,
    layout: RuntimeLayout,
    command: ReleasedCommand,
    lang: &str,
    num_speakers: u32,
    input_kind: InputKind,
    inputs: &[std::path::PathBuf],
    out_dir: Option<&std::path::Path>,
    options: Option<&CommandOptions>,
    lexicon: Option<&str>,
    before: Option<&std::path::Path>,
    force_cpu: bool,
    allow_mps: bool,
    workers: Option<usize>,
    timeout: Option<u64>,
    sequential: bool,
) -> Result<(), CliError> {
    if let Some(workers) = workers {
        // `--workers N` is always an explicit override; `Some(_)`
        // wins over the host-facts recommendation.
        cfg.max_workers_per_job = Some(workers as u32);
    }
    if let Some(timeout) = timeout {
        cfg.audio_task_timeout_s = timeout;
    }
    if sequential {
        apply_sequential_config(&mut cfg);
    }

    let mapping_keys: Vec<String> = cfg
        .media_mappings
        .keys()
        .map(|k| k.as_str().to_owned())
        .collect();
    let Some(prepared) = prepare_paths_submission(
        command,
        lang,
        num_speakers,
        input_kind,
        inputs,
        out_dir,
        options,
        lexicon,
        before,
        &mapping_keys,
    )?
    else {
        eprintln!("warning: no files found for {input_kind:?} input");
        return Ok(());
    };

    eprintln!("Found {} file(s) to process.\n", prepared.total_files);
    crate::cli::discover::print_passthrough_skips(&prepared.passthrough);
    eprintln!("Running locally (direct mode)...\n");

    let direct_workers = prepare_workers(
        build_direct_pool_config(&cfg, force_cpu, allow_mps, layout.state_dir()),
        RegistryDiscovery::Ignore,
    )
    .await
    .map_err(CliError::from)?;
    let host = DirectHost::new(cfg, layout, None, None, &direct_workers)
        .await
        .map_err(CliError::from)?;
    let job_id = host
        .submit_submission(prepared.submission)
        .await
        .map_err(CliError::from)?;
    let submitted_debug = host
        .job_debug_artifacts(&job_id)
        .await
        .map_err(CliError::from)?;
    eprintln!("Direct job prepared.\n");
    helpers::print_job_debug_artifacts(&submitted_debug);
    eprintln!();

    let progress = BatchProgress::new(prepared.total_files as u64, command.as_wire_name());
    let mut tracker = DirectProgressTracker::default();
    if let Ok(initial) = host.job_info(&job_id).await {
        tracker.observe(&progress, &initial);
    }

    let runner_host = host.clone();
    let run_job_id = job_id.clone();
    let run_fut = async move {
        runner_host
            .run_job(&run_job_id)
            .await
            .map_err(CliError::from)
    };
    tokio::pin!(run_fut);
    let mut poll_interval = tokio::time::interval(Duration::from_millis(120));
    poll_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

    let final_info: JobInfo = loop {
        tokio::select! {
            result = &mut run_fut => {
                if let Err(error) = result {
                    progress.finish();
                    return Err(error);
                }
                let info = match host.job_info(&job_id).await {
                    Ok(info) => info,
                    Err(error) => {
                        progress.finish();
                        return Err(CliError::from(error));
                    }
                };
                tracker.observe(&progress, &info);
                break info;
            }
            _ = poll_interval.tick() => {
                if let Ok(info) = host.job_info(&job_id).await {
                    tracker.observe(&progress, &info);
                }
            }
        }
    };
    progress.finish();

    let error_details = file_error_details(&final_info);
    let clean_success = final_info.status == JobStatus::Completed
        && error_details.is_empty()
        && final_info
            .error
            .as_ref()
            .is_none_or(|s| s.trim().is_empty());
    if !clean_success {
        match host.job_debug_artifacts(&job_id).await {
            Ok(artifacts) => {
                eprintln!();
                helpers::print_job_debug_artifacts(&artifacts);
                eprintln!();
            }
            Err(error) => eprintln!("warning: failed to collect direct debug artifacts: {error}"),
        }
    }
    finish_terminal_job(
        &final_info,
        &error_details,
        prepared.total_files as u64,
        &prepared.effective_out,
    )
}

pub(crate) fn build_direct_pool_config(
    cfg: &ServerConfig,
    force_cpu: bool,
    allow_mps: bool,
    state_dir: &std::path::Path,
) -> PoolConfig {
    let tier = cfg.resolved_memory_tier();
    let host_policy = HostExecutionPolicy::from_server_config(cfg);
    // Same boundary conversion as `serve_cmd::start`: CLI
    // `--force-cpu` becomes `Some(true)`; absent leaves the YAML
    // value (default `None`) intact for the recommendation to fill.
    let mut cfg_for_resolve = cfg.clone();
    if force_cpu {
        cfg_for_resolve.force_cpu = Some(true);
    }
    if allow_mps {
        cfg_for_resolve.allow_mps = Some(true);
    }
    let effective = EffectiveConfig::resolve_from_server_config(&cfg_for_resolve);
    let worker_runtime = WorkerRuntimeConfig {
        force_cpu: effective.force_cpu,
        allow_mps: effective.allow_mps,
        gpu_thread_pool_size: effective.gpu_thread_pool_size,
        host_memory: HostMemoryRuntimeConfig::from_server_config(cfg),
        memory_tier: tier,
        bootstrap_mode: host_policy.bootstrap_mode,
        ..WorkerRuntimeConfig::default()
    };
    PoolConfig {
        python_path: resolve_python_executable(),
        health_check_interval_s: if cfg.worker_health_interval_s > 0 {
            cfg.worker_health_interval_s
        } else {
            PoolConfig::default().health_check_interval_s
        },
        verbose: 0,
        // See serve_cmd: the state dir comes from the resolved RuntimeLayout,
        // never inferred from worker_registry_path.
        runtime: worker_runtime.with_state_dir(state_dir.to_path_buf()),
        max_workers_per_key: match cfg.max_workers_per_key {
            Some(n) => crate::host_facts::PerProfile::uniform(n as usize),
            None => effective.max_workers_per_key_by_profile.map(|n| n as usize),
        },
        ready_timeout_s: if cfg.worker_ready_timeout_s > 0 {
            cfg.worker_ready_timeout_s
        } else {
            PoolConfig::default().ready_timeout_s
        },
        max_total_workers: effective.max_total_workers as usize,
        audio_task_timeout_s: cfg.audio_task_timeout_s,
        analysis_task_timeout_s: cfg.analysis_task_timeout_s,
        ensure_task_timeout_s: cfg.ensure_task_timeout_s,
        worker_registry_path: cfg.worker_registry_path.clone(),
        ..PoolConfig::default()
    }
}

/// Whether the server advertises `command`. The server's list holds released
/// command names only (a test-echo worker advertises the commands it echoes,
/// not a name of its own), so the command's own name is the only match.
fn server_supports_command(capabilities: &[String], command: ReleasedCommand) -> bool {
    capabilities.is_empty() || capabilities.iter().any(|c| c == command.as_str())
}

/// Refuse a server whose build identity is not this CLI's, before any job is
/// submitted and so before any compute runs.
///
/// A server left running across an upgrade produces results with the older
/// build's engines and fixes, and the CLI writes them as if the current build
/// ran. That used to be a warning printed above output that was produced
/// anyway. A server that reports no build identity is refused too: nothing
/// then shows it is this build. The remedy is in the error: restart the
/// server with this build.
fn refuse_foreign_server_build(
    server_url: &str,
    health: &crate::api::HealthResponse,
) -> Result<(), CliError> {
    let client_build = crate::build_hash();
    let server_build = match health.build_hash.as_str() {
        "" => crate::cli::error::ServerBuild::Unreported,
        reported if reported == client_build => return Ok(()),
        reported => crate::cli::error::ServerBuild::Reported(reported.to_owned()),
    };
    Err(CliError::ServerBuildMismatch {
        // The full URL, not the short label: the operator has to find and
        // restart exactly this server.
        server: server_url.to_owned(),
        server_build,
        client_build: client_build.to_owned(),
    })
}

/// Probe the local server with a short timeout.
///
/// Returns a human-readable status string if the server is healthy,
/// `None` if unreachable or unhealthy. Uses a 500ms timeout so direct
/// mode startup is not noticeably delayed when no server is running.
async fn probe_local_server(url: &str) -> Option<String> {
    let health_url = format!("{url}/health");
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_millis(500))
        .build()
        .ok()?;
    let resp = client.get(&health_url).send().await.ok()?;
    // Parsing IS the identification: `status` and `version` are required, so a
    // body that is not a Batchalign health response fails here rather than
    // arriving with defaults that agree with whatever was asked of it.
    let health: crate::api::HealthResponse = resp.json().await.ok()?;
    // Exhaustive: a future non-`Ok` variant must be decided here, not
    // accepted. Why the `if` this replaces could never fire: see
    // `HealthResponse::status`.
    match health.status {
        crate::api::HealthStatus::Ok => {}
    }
    let label = if health.active_jobs > 0 {
        format!(
            "{} workers, {} active job(s)",
            health.workers_available, health.active_jobs
        )
    } else {
        format!("{} workers available", health.workers_available)
    };
    Some(label)
}

/// Apply `--sequential` overrides to a server config: disable memory gate,
/// cap workers to 1 per key. Eviction is now pressure-driven; the
/// single worker stays loaded until host memory pressure forces an
/// eviction (which never fires under sequential synthetic-test loads
/// on a developer machine).
pub(crate) fn apply_sequential_config(cfg: &mut ServerConfig) {
    use crate::api::MemoryMb;
    // `Some(MemoryMb(1))` effectively disables the gate, the
    // coordinator requires only 1 MB free, which is always true.
    cfg.memory_gate_mb = Some(MemoryMb(1));
    cfg.max_workers_per_key = Some(1);
    cfg.max_concurrent_worker_startups = 1;
}

#[cfg(test)]
mod tests {
    use crate::api::MemoryMb;
    use crate::config::ServerConfig;
    use crate::{ReleasedCommand, released_command_uses_local_audio};

    use super::*;

    #[test]
    fn benchmark_and_align_are_treated_as_local_audio_commands() {
        assert!(released_command_uses_local_audio(
            ReleasedCommand::Benchmark
        ));
        assert!(released_command_uses_local_audio(
            ReleasedCommand::Transcribe
        ));
        assert!(released_command_uses_local_audio(ReleasedCommand::Align));
        assert!(!released_command_uses_local_audio(
            ReleasedCommand::Morphotag
        ));
    }

    /// `--sequential` effectively disables the memory gate (threshold = 1 MB).
    /// `Some(MemoryMb(1))` keeps the resolved value at 1 MB; `None`
    /// would fall through to the tier-derived headroom and so would
    /// not effectively disable.
    #[test]
    fn sequential_config_disables_memory_gate() {
        let mut cfg = ServerConfig::default();
        // Default is `None`: resolved value comes from the tier
        // headroom, which is a real positive number on any host that
        // can run this test.
        assert_eq!(
            cfg.memory_gate_mb, None,
            "precondition: default leaves the gate at the tier-derived value"
        );
        assert!(
            cfg.resolved_memory_gate_mb().0 > 1,
            "precondition: resolved gate is enabled by default"
        );
        apply_sequential_config(&mut cfg);
        assert_eq!(cfg.memory_gate_mb, Some(MemoryMb(1)));
    }

    /// `--sequential` caps workers to 1 per key.
    #[test]
    fn sequential_config_caps_workers_to_one() {
        let mut cfg = ServerConfig::default();
        apply_sequential_config(&mut cfg);
        assert_eq!(cfg.max_workers_per_key, Some(1));
        assert_eq!(cfg.max_concurrent_worker_startups, 1);
    }

    /// Answer one HTTP request on an ephemeral loopback port with a canned
    /// body, and hand back the URL.
    ///
    /// A real listener rather than a mocked client, because the defect was in
    /// what the probe ACCEPTS off the wire: a double that returned a
    /// `HealthResponse` would have modelled the bug out of existence.
    async fn serve_canned(status_line: &str, body: &'static str) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind an ephemeral loopback port");
        let port = listener.local_addr().expect("local addr").port();
        let status_line = status_line.to_string();
        tokio::spawn(async move {
            // One request is all the probe makes.
            if let Ok((mut socket, _)) = listener.accept().await {
                use tokio::io::{AsyncReadExt, AsyncWriteExt};
                let mut buf = [0u8; 2048];
                let _ = socket.read(&mut buf).await;
                let response = format!(
                    "{status_line}\r\nContent-Type: application/json\r\n\
                     Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = socket.write_all(response.as_bytes()).await;
                let _ = socket.flush().await;
            }
        });
        format!("http://127.0.0.1:{port}")
    }

    /// A foreign HTTP process on the loopback port is NOT a Batchalign server.
    ///
    /// Measured on 2026-08-27: an unrelated local service was listening on the
    /// default port 8000 and answering every path with
    /// `{"error":{"message":"unknown endpoint",...}}`. `batchalign3 align`
    /// printed `Using local server at http://127.0.0.1:8000 (0 workers
    /// available)` and dispatched the job to it.
    ///
    /// The cause was not this function's logic but the TYPE it parsed into;
    /// the account is on `HealthResponse::status`. A probe whose output is
    /// identical whether or not the thing being probed exists is not evidence,
    /// which is why the assertion below is about REJECTION and there is a
    /// control test beside it.
    #[tokio::test]
    async fn a_foreign_http_process_is_not_accepted_as_a_local_server() {
        let url = serve_canned(
            "HTTP/1.1 404 Not Found",
            r#"{"error":{"message":"unknown endpoint","type":"invalid_request_error"}}"#,
        )
        .await;

        assert!(
            probe_local_server(&url).await.is_none(),
            "a process that answers /health with a foreign body was accepted \
             as a Batchalign server; jobs would be dispatched to it"
        );
    }

    /// The control: a real server must still be accepted.
    ///
    /// The cheapest way to pass the test above is to reject everything, which
    /// would silently disable local-server dispatch for every user.
    #[tokio::test]
    async fn a_real_batchalign_health_response_is_still_accepted() {
        let url = serve_canned(
            "HTTP/1.1 200 OK",
            r#"{"status":"ok","version":"0.2.0","node_id":"test-node",
                "free_threaded":false,"capabilities":["align"],
                "loaded_pipelines":[],"media_roots":[],"media_mapping_keys":[],
                "workers_available":2,"active_jobs":0}"#,
        )
        .await;

        assert!(
            probe_local_server(&url).await.is_some(),
            "a genuine Batchalign health response was rejected"
        );
    }

    /// `--sequential` + `--server` is rejected.
    #[tokio::test]
    async fn sequential_rejects_server_flag() {
        let result = dispatch(
            DispatchRequest {
                command: ReleasedCommand::Morphotag,
                lang: "eng",
                num_speakers: 0,
                input_kind: InputKind::Chat,
                server_arg: Some("http://server-01:8001"),
                inputs: &[],
                out_dir: None,
                options: None,
                bank: None,
                subdir: None,
                lexicon: None,
                use_tui: false,
                open_dashboard: false,
                force_cpu: false,
                allow_mps: false,
                no_server: false,
                before: None,
                workers: None,
                timeout: None,
                sequential: true,
                memory_tier: None,
            },
            // Deliberately a path that does not exist: the mutual-exclusion
            // check runs before any config is loaded, so a layout this test
            // could actually read would imply a dependency it does not have.
            &RuntimeLayout::from_state_dir(std::path::PathBuf::from(
                "/nonexistent/dispatch-arg-validation",
            )),
        )
        .await;
        let err = result.expect_err("should reject --sequential --server");
        assert!(
            err.to_string().contains("mutually exclusive"),
            "error should mention mutual exclusivity: {err}"
        );
    }
}
