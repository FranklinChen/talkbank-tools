//! Dispatch router: routes processing commands to direct or server hosts.
//!
//! Mirrors `dispatch.py` + `dispatch_server.py`.
//!
//! Explicit `--server` always selects that HTTP server. Without an explicit
//! target, local processing commands can auto-spawn a daemon or execute inline
//! through the shared direct host.

mod display;
mod helpers;
mod paths;
mod plan;
mod single;

use std::time::Duration;

use crate::config::{RuntimeLayout, load_config_from_layout};
use crate::options::CommandOptions;
use crate::worker::handle::WorkerRuntimeConfig;
use crate::worker::pool::PoolConfig;
use crate::{DirectHost, RegistryDiscovery, ReleasedCommand, prepare_workers};
use crate::{api::JobInfo, api::JobStatus, api::LanguageSpec, config::ServerConfig};

use crate::cli::client::{self, BatchalignClient};
use crate::cli::daemon::{self, RequestedRuntime, RuntimeRefusal};
use crate::cli::discover::PassthroughMode;
use crate::cli::error::CliError;
use crate::cli::output::Overwrite;
use crate::cli::progress::BatchProgress;
use crate::cli::python::resolve_python_executable;

use crate::cli::args::{Dashboard, InputKind, Progress};
use crate::cli::daemon::ServerOrigin;
use display::{BrowserEnv, Session, dashboard_launch};
use helpers::{DirectProgressTracker, file_error_details, finish_terminal_job};
use paths::prepare_paths_submission;
pub use plan::Execution;
use single::{ServerTarget, dispatch_single_server};

// ---------------------------------------------------------------------------
// Top-level dispatch router
// ---------------------------------------------------------------------------

/// Where the user asked the job to run: `--server URL`, `--no-server`,
/// `--sequential`, or none of them.
///
/// Three flags encode one closed choice, and two of their combinations are
/// meaningless (`--sequential` with `--server`, refused) or redundant
/// (`--sequential` implies `--no-server`). [`ServerRequest::decide`] is the
/// one place that reads the raw flags; the run and the `--dry-run` plan both
/// route on its value, so the plan cannot route differently from the run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ServerRequest<'a> {
    /// `--server URL`: that server and no other (falling through to a local
    /// daemon after the operator named another server would send paid work
    /// to the wrong process).
    Named(&'a str),
    /// No server was named: this host decides.
    Local(LocalRoute),
}

/// How a job with no named server runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum LocalRoute {
    /// The managed local daemon (started if `auto_daemon` allows), else a
    /// loopback server already running, else "no server available".
    Managed,
    /// `--no-server` or `--sequential`: this process runs the job itself.
    Direct(DirectPace),
}

/// The pace of a job this process runs itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DirectPace {
    /// `--no-server`: the worker pool at its configured size.
    Pooled,
    /// `--sequential`: one file at a time, with the memory gate relaxed.
    Sequential,
}

impl<'a> ServerRequest<'a> {
    /// Read the three routing flags once. A named server with `--sequential`
    /// or `--no-server` is refused: a named server runs at its own settings,
    /// so "one at a time" cannot be honoured there, and "no server" contradicts
    /// it. (`--server` may come from `BATCHALIGN_SERVER`, so the refusal names
    /// the variable too; before 2026-11-04 `--no-server` was silently ignored
    /// beside it.)
    pub(super) fn decide(
        server_arg: Option<&'a str>,
        no_server: bool,
        sequential: bool,
    ) -> Result<Self, CliError> {
        match (server_arg, no_server, sequential) {
            (Some(_), _, true) => Err(CliError::InvalidArgument(
                "--sequential and --server are mutually exclusive".into(),
            )),
            (Some(_), true, false) => Err(CliError::InvalidArgument(
                "--no-server contradicts --server (or BATCHALIGN_SERVER): \
                 drop one of them"
                    .into(),
            )),
            (Some(server), false, false) => Ok(Self::Named(server)),
            (None, _, true) => Ok(Self::Local(LocalRoute::Direct(DirectPace::Sequential))),
            (None, true, false) => Ok(Self::Local(LocalRoute::Direct(DirectPace::Pooled))),
            (None, false, false) => Ok(Self::Local(LocalRoute::Managed)),
        }
    }
}

/// Named dispatch request for one CLI processing invocation.
#[derive(Debug)]
pub struct DispatchRequest<'a> {
    /// Job-level language for the command, already typed at the CLI.
    pub lang: LanguageSpec,
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
    /// Typed command options for submission. They also name the command:
    /// there is no separate command field to disagree with them.
    pub options: CommandOptions,
    /// Optional TalkBank bank name.
    pub bank: Option<&'a str>,
    /// Optional bank subdirectory.
    pub subdir: Option<&'a str>,
    /// Optional lexicon path.
    pub lexicon: Option<&'a str>,
    /// How a server-backed job's progress is shown (`--tui` / `--no-tui`).
    pub progress: Progress,
    /// Whether the job's dashboard page opens (`--open-dashboard` /
    /// `--no-open-dashboard`, or once on the invocation that starts the daemon).
    pub dashboard: Dashboard,
    /// Whether to force CPU execution for local worker processes.
    pub force_cpu: bool,
    /// Explicit Apple-GPU opt-in (`--allow-mps`), threaded to workers.
    pub allow_mps: bool,
    /// Skip auto-detection of a local server (force direct mode).
    pub no_server: bool,
    /// Optional before-path input for incremental workflows.
    pub before: Option<&'a std::path::Path>,
    /// `--workers` and `--timeout`, as passed: honoured by a daemon this
    /// invocation starts or by direct execution, held against a reused
    /// daemon's record, and refused by a server that records neither.
    pub runtime: RequestedRuntime,
    /// Sequential mode: one worker, no memory gate, no server.
    pub sequential: bool,
    /// Override auto-detected memory tier (small, medium, large, fleet).
    pub memory_tier: Option<crate::types::runtime::MemoryTierKind>,
    /// `--overwrite`: authority to replace a generated result file that already
    /// exists beside its input. Absent, such a destination is refused before
    /// submission (see `ResultDestinations::admit`).
    pub overwrite: Option<Overwrite>,
    /// Submit the job, or (`--dry-run`) plan and print it and stop.
    pub execution: Execution,
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
        progress,
        dashboard,
        force_cpu,
        allow_mps,
        no_server,
        before,
        runtime,
        sequential,
        memory_tier,
        overwrite,
        execution,
    } = request;
    let command = options.command();
    // Two facts about this process, read once: whether a person is at the
    // terminal, and whether a harness forbade the browser. Each server-backed
    // path below decides the dashboard from them and from how it came by its
    // server.
    let session = Session::of_stdout();
    let browser = BrowserEnv::of_env();

    let server = ServerRequest::decide(server_arg, no_server, sequential)?;
    // `--sequential` is a cap of one, over whatever `--workers` said.
    let runtime = match server {
        ServerRequest::Local(LocalRoute::Direct(DirectPace::Sequential)) => RequestedRuntime {
            workers: Some(crate::api::WorkerCap::literal::<1>()),
            timeout: runtime.timeout,
        },
        ServerRequest::Named(_)
        | ServerRequest::Local(LocalRoute::Managed | LocalRoute::Direct(DirectPace::Pooled)) => {
            runtime
        }
    };

    if bank.is_some() || subdir.is_some() {
        eprintln!(
            "error: --bank/--subdir remote media selection is no longer supported.\n\
             Pass filesystem paths that are visible on the execution host instead."
        );
        return Ok(());
    }

    // `--dry-run`: everything up to submission, then the plan, then nothing.
    // Decided here, after the argument checks every run makes and before any
    // route can start a daemon, open a pool or submit.
    if let Execution::PlanOnly { unresolved_inputs } = execution {
        let plan = plan::build_plan(
            plan::PlanRequest {
                lang: &lang,
                num_speakers,
                input_kind,
                server,
                inputs,
                out_dir,
                options: &options,
                lexicon,
                before,
                runtime,
                memory_tier,
                overwrite,
                unresolved_inputs,
            },
            layout,
        )
        .await?;
        println!("{plan}");
        return Ok(());
    }

    // 1. An explicit --server always identifies the execution host. Loopback
    // targets can safely send filesystem paths; other targets use content and
    // media-reference submission. Falling through to a healthy local daemon
    // after the operator named another server makes the producer identity
    // false and can send paid work to the wrong process.
    let local = match server {
        ServerRequest::Local(local) => local,
        ServerRequest::Named(server) => {
            let client = BatchalignClient::new()?;
            let urls = client::parse_servers(server);
            if urls.is_empty() {
                eprintln!("error: no server URL provided");
                return Ok(());
            }

            if urls.len() == 1 {
                // A named server runs with the settings it was started with, and
                // nothing here records them, so `--workers`/`--timeout` beside
                // `--server` can only be refused, never applied or silently dropped.
                RuntimeRefusal::admit_unrecorded(runtime, &urls[0])?;
                let target = ServerTarget::parse_explicit(&urls[0], command)?;
                return dispatch_single_server(
                    &client,
                    &target,
                    &lang,
                    num_speakers,
                    input_kind,
                    inputs,
                    out_dir,
                    &options,
                    lexicon,
                    before,
                    progress,
                    session,
                    // A named server was running before this invocation looked.
                    dashboard_launch(dashboard, ServerOrigin::AlreadyRunning, session, browser),
                    overwrite,
                )
                .await;
            }

            eprintln!(
                "error: multi-server dispatch (--server URL1,URL2) is not available in this version.\n\
             Use --server with a single URL instead."
            );
            return Ok(());
        }
    };

    let mut cfg = load_config_from_layout(layout, None)?;

    // Apply --memory-tier CLI override (for testing constrained-memory behavior
    // on large machines). Overrides auto-detection from system RAM.
    if let Some(tier) = memory_tier {
        cfg.memory_tier = Some(tier);
    }

    // A job this process runs itself skips every server step below.
    let pace = match local {
        LocalRoute::Direct(pace) => pace,
        LocalRoute::Managed => {
            // 2. Auto-daemon routing applies only when no server was explicitly named.
            let local_daemon = if cfg.auto_daemon {
                daemon::ensure_daemon(
                    layout,
                    &cfg,
                    daemon::DaemonRequest {
                        force_cpu,
                        allow_mps,
                        runtime,
                    },
                )
                .await?
            } else {
                None
            };

            if let Some(local_daemon) = local_daemon {
                let local_daemon_url = local_daemon.url.as_str();
                eprintln!("Submitting to local daemon at {local_daemon_url}\n");
                let client = BatchalignClient::new()?;
                let target = ServerTarget::parse_shared_filesystem(local_daemon_url)?;
                return dispatch_single_server(
                    &client,
                    &target,
                    &lang,
                    num_speakers,
                    input_kind,
                    inputs,
                    out_dir,
                    &options,
                    lexicon,
                    before,
                    progress,
                    session,
                    // The one path that can have started the daemon: this is where the
                    // default opens the dashboard, once.
                    dashboard_launch(dashboard, local_daemon.origin, session, browser),
                    overwrite,
                )
                .await;
            }

            // 3. Auto-detect a loopback server that was started outside the daemon
            // state-file flow (for example by launchd or a foreground serve command).
            // Prefer the port the server published over the one the config asked for.
            // Reached only on the managed route: `--no-server` and `--sequential` never
            // contact a server. `cfg` is already loaded, so this costs one small file read.
            if let Some(local_url) = daemon::local_server_url(layout, cfg.port)
                && let Some(health) = probe_local_server(&local_url).await
            {
                // Started outside the daemon flow, so nothing records its settings.
                RuntimeRefusal::admit_unrecorded(runtime, &local_url)?;
                eprintln!(
                    "Using local server at {local_url} ({})\n",
                    health_label(&health)
                );
                let client = BatchalignClient::new()?;
                let target = ServerTarget::parse_shared_filesystem(&local_url)?;
                return dispatch_single_server(
                    &client,
                    &target,
                    &lang,
                    num_speakers,
                    input_kind,
                    inputs,
                    out_dir,
                    &options,
                    lexicon,
                    before,
                    progress,
                    session,
                    // Found running, started by something else.
                    dashboard_launch(dashboard, ServerOrigin::AlreadyRunning, session, browser),
                    overwrite,
                )
                .await;
            }

            // 4. The managed route found no server, and the user did not ask for
            //    offline execution. Exit gracefully and create the output directory
            //    so tools that check for its existence still see a consistent
            //    post-run filesystem state.
            if let Some(dir) = out_dir {
                let _ = std::fs::create_dir_all(dir);
            }
            eprintln!("no server available");
            return Ok(());
        }
    };

    // Direct local execution (--no-server / --sequential path)
    dispatch_direct_mode(
        cfg,
        layout.clone(),
        &lang,
        num_speakers,
        input_kind,
        inputs,
        out_dir,
        &options,
        lexicon,
        before,
        force_cpu,
        allow_mps,
        runtime,
        pace,
        overwrite,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
fn dispatch_direct_mode<'run>(
    mut cfg: ServerConfig,
    layout: RuntimeLayout,
    lang: &'run LanguageSpec,
    num_speakers: u32,
    input_kind: InputKind,
    inputs: &'run [std::path::PathBuf],
    out_dir: Option<&'run std::path::Path>,
    options: &'run CommandOptions,
    lexicon: Option<&'run str>,
    before: Option<&'run std::path::Path>,
    force_cpu: bool,
    allow_mps: bool,
    runtime: RequestedRuntime,
    pace: DirectPace,
    overwrite: Option<Overwrite>,
) -> futures::future::BoxFuture<'run, Result<(), CliError>> {
    Box::pin(async move {
        // `--workers N` and `--timeout S` are explicit overrides of the
        // configured (or host-facts recommended) values.
        runtime.apply_to(&mut cfg);
        match pace {
            DirectPace::Sequential => apply_sequential_config(&mut cfg),
            DirectPace::Pooled => {}
        }

        let mapping_keys: Vec<String> = cfg
            .media_mappings
            .keys()
            .map(|k| k.as_str().to_owned())
            .collect();
        let command = options.command();
        let Some(prepared) = prepare_paths_submission(
            options,
            lang,
            num_speakers,
            input_kind,
            inputs,
            out_dir,
            lexicon,
            before,
            &mapping_keys,
            overwrite,
            PassthroughMode::Copy,
        )?
        else {
            eprintln!("warning: no files found for {input_kind:?} input");
            return Ok(());
        };

        eprintln!("Found {} file(s) to process.\n", prepared.total_files);
        crate::cli::discover::print_passthrough_skips(&prepared.passthrough);
        eprintln!("Running locally (direct mode)...\n");

        // One reading of the host for this run, with the CLI's device switches
        // applied: the pool and every job's dispatch context resolve from it.
        let cfg = cfg.with_device_flags(force_cpu, allow_mps);
        let host_snapshot = std::sync::Arc::new(crate::host_facts::HostSnapshot::read(&cfg)?);
        let direct_workers = prepare_workers(
            host_snapshot.clone(),
            build_direct_pool_config(&cfg, &host_snapshot, layout.state_dir()),
            RegistryDiscovery::Ignore,
        )
        .await
        .map_err(CliError::from)?;
        let host = DirectHost::new(
            cfg,
            layout,
            None,
            None,
            &direct_workers,
            std::sync::Arc::new(crate::clock::SystemClock),
        )
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
                Err(error) => {
                    eprintln!("warning: failed to collect direct debug artifacts: {error}")
                }
            }
        }
        finish_terminal_job(
            &final_info,
            &error_details,
            prepared.total_files as u64,
            &prepared.destinations.reporting_directory()?,
            crate::cli::error::DiagnosedRelist::ThisRunOnly,
        )
    })
}

/// The direct-mode pool for `cfg` (device switches already applied) on the
/// host as read. Every field is stated: a production pool is never built on
/// a base of defaults, so a new `PoolConfig` field breaks this literal until
/// it is decided here, instead of silently taking a test value.
pub(crate) fn build_direct_pool_config(
    cfg: &ServerConfig,
    host: &crate::host_facts::HostSnapshot,
    state_dir: &std::path::Path,
) -> PoolConfig {
    let effective = host.effective();
    let worker_runtime = WorkerRuntimeConfig {
        force_cpu: effective.force_cpu,
        allow_mps: effective.allow_mps,
        gpu_thread_pool_size: effective.gpu_thread_pool_size,
        bootstrap_mode: host.policy().bootstrap_mode,
        ..WorkerRuntimeConfig::local(host)
    };
    PoolConfig {
        python_path: resolve_python_executable(),
        health_check_interval_s: cfg.worker_health_interval_s,
        startup_stall: cfg.worker_startup_stall(),
        test_echo: false,
        max_workers_per_key: match cfg.max_workers_per_key {
            Some(n) => crate::host_facts::PerProfile::uniform(n as usize),
            None => effective.max_workers_per_key_by_profile.map(|n| n as usize),
        },
        max_total_workers: effective.max_total_workers as usize,
        checkout_wait_report_interval: None,
        verbose: 0,
        // See serve_cmd: the state dir comes from the resolved RuntimeLayout,
        // never inferred from worker_registry_path.
        runtime: worker_runtime.with_state_dir(state_dir.to_path_buf()),
        task_timeouts: cfg.task_timeouts(),
        ensure_task_stall: cfg.ensure_task_stall(),
        worker_registry_path: cfg.worker_registry_path.clone(),
        test_delay_ms: 0,
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
/// Returns the server's health if it answered as a Batchalign server, `None`
/// if unreachable or unhealthy. Uses a 500ms timeout so direct mode startup
/// is not noticeably delayed when no server is running.
async fn probe_local_server(url: &str) -> Option<crate::api::HealthResponse> {
    let health_url = format!("{url}/health");
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_millis(500))
        .build()
        .ok()?;
    let resp = client.get(&health_url).send().await.ok()?;
    if !resp.status().is_success() {
        return None;
    }
    // Parsed as the typed response, so a process that answers 200 with a
    // body that is not a Batchalign health response fails here rather than
    // arriving with defaults that agree with whatever was asked of it.
    let health: crate::api::HealthResponse = resp.json().await.ok()?;
    // Exhaustive: a future non-`Ok` variant must be decided here, not
    // accepted. Why the `if` this replaces could never fire: see
    // `HealthResponse::status`.
    match health.status {
        crate::api::HealthStatus::Ok => {}
    }
    Some(health)
}

/// What a health response says about a server's capacity, as the CLI prints
/// it when it submits to that server or plans to.
fn health_label(health: &crate::api::HealthResponse) -> String {
    if health.active_jobs > 0 {
        format!(
            "{} workers, {} active job(s)",
            health.workers_available, health.active_jobs
        )
    } else {
        format!("{} workers available", health.workers_available)
    }
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
    cfg.max_concurrent_worker_startups = crate::config::WorkerStartupLimit::literal::<1>();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::MemoryMb;
    use crate::config::ServerConfig;

    /// Every combination of the three routing flags lands on exactly one
    /// route or one named refusal; none is silently ignored.
    #[test]
    fn the_routing_flags_decide_one_route_or_a_named_refusal() {
        use super::{DirectPace, LocalRoute, ServerRequest};
        let url = "http://127.0.0.1:9";
        let decide = ServerRequest::decide;
        assert_eq!(
            decide(None, false, false).unwrap(),
            ServerRequest::Local(LocalRoute::Managed)
        );
        assert_eq!(
            decide(None, true, false).unwrap(),
            ServerRequest::Local(LocalRoute::Direct(DirectPace::Pooled))
        );
        // `--sequential` implies running here, with or without `--no-server`.
        for no_server in [false, true] {
            assert_eq!(
                decide(None, no_server, true).unwrap(),
                ServerRequest::Local(LocalRoute::Direct(DirectPace::Sequential))
            );
        }
        assert_eq!(
            decide(Some(url), false, false).unwrap(),
            ServerRequest::Named(url)
        );
        // A named server beside "run here" is refused, never silently preferred.
        let contradiction = decide(Some(url), true, false).unwrap_err().to_string();
        assert!(contradiction.contains("--no-server"), "{contradiction}");
        assert!(
            contradiction.contains("BATCHALIGN_SERVER"),
            "{contradiction}"
        );
        for no_server in [false, true] {
            let refused = decide(Some(url), no_server, true).unwrap_err().to_string();
            assert!(refused.contains("--sequential and --server"), "{refused}");
        }
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
        assert_eq!(cfg.max_concurrent_worker_startups.get(), 1);
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
                lang: LanguageSpec::Resolved(crate::api::LanguageCode3::eng()),
                num_speakers: 0,
                input_kind: InputKind::Chat,
                server_arg: Some("http://server-01:8001"),
                inputs: &[],
                out_dir: None,
                options: CommandOptions::Morphotag(crate::options::MorphotagOptions::default()),
                bank: None,
                subdir: None,
                lexicon: None,
                progress: Progress::Plain,
                dashboard: Dashboard::Closed,
                force_cpu: false,
                allow_mps: false,
                no_server: false,
                before: None,
                runtime: RequestedRuntime::NONE,
                sequential: true,
                memory_tier: None,
                overwrite: None,
                execution: Execution::Submit,
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
