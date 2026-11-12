//! `batchalign3 serve` -- manage the batchalign processing server.
//!
//! This module implements the three `serve` subcommands:
//!
//! - **`serve start`** -- Launch the HTTP server that accepts processing jobs.
//!   In foreground mode (`--foreground`) the server runs in the current process,
//!   blocking until shutdown. In background mode (the default) a detached child
//!   process is spawned in a new session (`setsid`) so it survives CLI exit; it
//!   publishes a handshake naming its PID and the port it bound, which is what
//!   the CLI reads back. CLI flags (port, host, Python path, test-echo)
//!   override values from `server.yaml`.
//!
//! - **`serve stop`** -- Shut down any running server and local daemon. Reads the
//!   published handshake, sends `SIGTERM` to the process group, and cleans up
//!   state files.
//!
//! - **`serve status`** -- Probe a running server's `/health` endpoint and print
//!   version, worker count, active jobs, and media root configuration. Discovers
//!   the server URL from `--server`, a local daemon info file, or falls back to
//!   the configured local server URL.

use crate::config::{self, RuntimeLayout};
use crate::server_handshake::{HandshakeSlot, ServerHandshake};
use crate::worker::handle::WorkerRuntimeConfig;
use crate::worker::pool::PoolConfig;

use crate::cli::args::{ServeStartArgs, ServeStatusArgs, ServeStatusFormat};
use crate::cli::client::BatchalignClient;
use crate::cli::daemon;
use crate::cli::error::CliError;
use crate::cli::python::resolve_python_executable;
use crate::cli::self_exe::resolve_self_exe;

/// `serve start`: start the processing server.
pub async fn start(
    args: &ServeStartArgs,
    verbose: u8,
    force_cpu: bool,
    allow_mps: bool,
) -> Result<(), CliError> {
    let layout = RuntimeLayout::from_env();
    let mut cfg =
        config::load_config_from_layout(&layout, args.config.as_deref().map(std::path::Path::new))?;
    let worker_python = args
        .python
        .clone()
        .unwrap_or_else(resolve_python_executable);

    // Override config values only when explicitly passed via CLI.
    if let Some(port) = args.port {
        // `--port 0` on the command line asks for an ephemeral bind, the same
        // as writing `port: 0` in the config.
        cfg.port = crate::config::PortRequest::from_u16(port);
    }
    if let Some(ref host) = args.host {
        cfg.host = host.clone();
    }
    // `--workers N` and `--timeout S` are explicit overrides of the config.
    args.runtime().apply_to(&mut cfg);

    if cfg.media_roots.is_empty() && cfg.media_mappings.is_empty() {
        eprintln!(
            "warning: no media_roots or media_mappings configured. \
             Align/transcribe commands will fail unless CHAT files reference \
             accessible media paths."
        );
    }

    if args.foreground {
        // The CLI's device switches become explicit overrides
        // (`ServerConfig::with_device_flags`), then the host is read once:
        // the tier, policy and resolved knobs printed and used below, and the
        // server's validation and every job's dispatch, all come from this
        // reading. The server does not start on a host it cannot read.
        let cfg = cfg.with_device_flags(force_cpu, allow_mps);
        let host = std::sync::Arc::new(crate::host_facts::HostSnapshot::read(&cfg)?);
        let tier = host.tier();
        let host_policy = host.policy();
        eprintln!(
            "\nStarting server on {}:{}...",
            cfg.host,
            cfg.port.describe()
        );
        eprintln!("Backend: local");
        eprintln!(
            "Memory tier: {}{} (total: {} GB, headroom: {} GB, stanza: {} GB, gpu: {} GB, bootstrap: {:?})\n",
            tier.kind,
            if cfg.memory_tier.is_some() {
                " (override)"
            } else {
                ""
            },
            tier.total_mb / 1000,
            tier.headroom_mb.0 / 1000,
            tier.stanza_startup_mb.0 / 1000,
            tier.gpu_startup_mb.0 / 1000,
            host_policy.bootstrap_mode,
        );

        // The runtime forms below remain concrete `u32` / `usize` values
        // because every downstream consumer (worker spawn args,
        // dispatch_semaphore permit counts, TcpWorkerInfo) already expects a
        // single value, not an override+recommendation pair.
        let effective = host.effective();
        let worker_runtime = WorkerRuntimeConfig {
            force_cpu: effective.force_cpu,
            allow_mps: effective.allow_mps,
            gpu_thread_pool_size: effective.gpu_thread_pool_size,
            bootstrap_mode: host_policy.bootstrap_mode,
            ..WorkerRuntimeConfig::local(&host)
        };
        let pool_config = PoolConfig {
            python_path: worker_python.clone(),
            test_echo: args.test_echo,
            health_check_interval_s: cfg.worker_health_interval_s,
            verbose,
            // Children inherit the SAME state dir this server resolved. Do not
            // infer it from `worker_registry_path`: that is a free-form file
            // path, and BATCHALIGN_STATE_DIR selects the whole runtime layout
            // (jobs/, logs/, server.yaml), not just workers.json.
            runtime: worker_runtime.with_state_dir(layout.state_dir().to_path_buf()),
            // Per-profile cap. `Some(n)` from server.yaml is the
            // operator's uniform override applied to all three
            // profiles; otherwise we use the host-facts per-profile
            // recommendation already resolved into `EffectiveConfig`.
            max_workers_per_key: match cfg.max_workers_per_key {
                Some(n) => crate::host_facts::PerProfile::uniform(n as usize),
                None => effective.max_workers_per_key_by_profile.map(|n| n as usize),
            },
            startup_stall: cfg.worker_startup_stall(),
            // `recommend_max_total_workers` clamps to `[2, 32]` so the
            // `as usize` cast is always well-defined.
            max_total_workers: effective.max_total_workers as usize,
            checkout_wait_report_interval: None,
            task_timeouts: cfg.task_timeouts(),
            ensure_task_stall: cfg.ensure_task_stall(),
            worker_registry_path: cfg.worker_registry_path.clone(),
            test_delay_ms: 0,
        };
        crate::serve_with_runtime(
            cfg,
            host,
            pool_config,
            layout,
            args.handshake_slot,
            Some(crate::build_hash().to_string()),
        )
        .await?;
    } else {
        // Background mode: spawn self with --foreground
        let exe = resolve_self_exe();

        std::fs::create_dir_all(layout.state_dir())?;

        // Stop any existing server; whether one was running does not matter
        // here, but a failure to stop it does, and so does a handshake that
        // cannot be read: it may name a live server, and the new one would
        // publish over it.
        if let daemon::StopOutcome::Unreadable(unreadable) =
            daemon::stop_manual_server(&layout).await?
        {
            return Err(unreadable.into());
        }

        let log_path = layout.server_log_path();
        // Append mode: preserve previous server logs across restarts.
        let log_file = std::fs::OpenOptions::new()
            .append(true)
            .create(true)
            .open(&log_path)?;

        let mut cmd = std::process::Command::new(&exe);
        cmd.args([
            "serve",
            "start",
            "--foreground",
            "--handshake-slot",
            args.handshake_slot.as_arg(),
            "--port",
            // The wire form, NOT `describe()`: this is the child's argv.
            &cfg.port.bind_value().to_string(),
            "--host",
            &cfg.host,
        ]);
        if let Some(ref config_path) = args.config {
            cmd.args(["--config", config_path]);
        }
        cmd.args(["--python", &worker_python]);
        if args.test_echo {
            cmd.arg("--test-echo");
        }
        if force_cpu {
            cmd.arg("--force-cpu");
        }
        if allow_mps {
            cmd.arg("--allow-mps");
        }
        // Forward verbosity to the background server process.
        for _ in 0..verbose {
            cmd.arg("-v");
        }
        // Forward `--workers`/`--timeout` to the background server process.
        cmd.args(args.runtime().serve_start_args());

        cmd.stdout(std::process::Stdio::null());
        cmd.stderr(log_file);

        // Start new session so it survives CLI exit
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            unsafe {
                cmd.pre_exec(|| {
                    libc::setsid();
                    Ok(())
                });
            }
        }

        // On Windows, CREATE_NEW_PROCESS_GROUP + DETACHED_PROCESS ensures
        // the server survives after the spawning CLI exits, analogous to
        // Unix setsid(). The server gets its own console group and is not
        // attached to the parent's console.
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            // CREATE_NEW_PROCESS_GROUP (0x200) | DETACHED_PROCESS (0x08)
            cmd.creation_flags(0x00000200 | 0x00000008);
        }

        let proc = cmd.spawn()?;
        let pid = proc.id();

        // Wait for the child to publish the port it BOUND, rather than
        // sleeping a fixed two seconds and then reporting the port we asked
        // for. The child is the only process that knows the answer, and for an
        // ephemeral request there is no answer to guess at.
        //
        // The parent deliberately writes nothing here. It used to record the
        // PID immediately, which under the new contract would race the child's
        // own record and could overwrite a published port with a
        // port-less one.
        let deadline = std::time::Instant::now() + crate::cli::daemon::startup_budget();
        let bound_port = match ServerHandshake::await_published(
            layout.state_dir(),
            args.handshake_slot,
            pid,
            deadline,
        )
        .await
        {
            Ok(port) => port,
            // The reason comes from the wait itself. Re-probing the process
            // here to guess it was racy: a server that timed out could exit
            // before the probe and be reported as having died immediately.
            Err(failure) => {
                let reason = failure.reason();
                eprintln!(
                    "\nerror: server process (PID {pid}) {reason}.\n\
                     Check the log file: {}\n\
                     hint: run `batchalign3 serve start --foreground` to see startup errors.",
                    log_path.display()
                );
                return Err(CliError::DaemonStartFailed);
            }
        };

        eprintln!("\nServer started (PID {pid})");
        eprintln!("Listening on http://{}:{bound_port}", cfg.host);
        eprintln!(
            "\nHandshake file: {}",
            ServerHandshake::path_in(layout.state_dir(), args.handshake_slot).display()
        );
        eprintln!("Log file: {}", log_path.display());
        eprintln!(
            "\nClients can now use: batchalign3 <command> ... \
             --server http://<this-machine>:{bound_port}"
        );
    }

    Ok(())
}

/// How many recorded processes `serve stop` stops, one after another.
const STOPPED_RECORDS: u32 = 3;

/// The longest `serve stop` can take: each recorded process in turn may use
/// its whole stop budget. The floor for a harness that kills the command on
/// a timeout (`tests/live_deadline`, `CliRunBudget::ServeStop`).
pub fn stop_budget() -> std::time::Duration {
    daemon::stop_budget_per_process() * STOPPED_RECORDS
}

/// `serve stop`: stop the server and daemon.
pub async fn stop() -> Result<(), CliError> {
    let layout = RuntimeLayout::from_env();

    // Daemons first, then the manually started server. Each outcome is
    // reported as what it was; a record that cannot be read is named and left
    // in place, and makes the command fail once everything else was tried.
    let outcomes: [_; STOPPED_RECORDS as usize] = [
        ("Local daemon", daemon::stop_daemon().await?),
        ("Sidecar daemon", daemon::stop_sidecar_daemon().await?),
        ("Server", daemon::stop_manual_server(&layout).await?),
    ];
    let mut unreadable = None;
    for (what, outcome) in outcomes {
        match outcome {
            daemon::StopOutcome::Stopped { pid } => eprintln!("{what} (PID {pid}) stopped."),
            daemon::StopOutcome::AlreadyDead { pid } => {
                eprintln!("{what} (PID {pid}) had already exited; its record was removed.")
            }
            daemon::StopOutcome::NotRunning => eprintln!("{what}: none recorded."),
            daemon::StopOutcome::Unreadable(record) => {
                eprintln!("{what}: {record}");
                unreadable.get_or_insert(record);
            }
        }
    }
    match unreadable {
        Some(record) => Err(record.into()),
        None => Ok(()),
    }
}

/// Machine-readable discovery never fabricates a ready URL from a failed probe.
#[derive(serde::Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
enum ServerObservation {
    Ready { url: String, build_hash: String },
    Stopped,
    Refused { message: String },
}

/// A configured address is not evidence of a running service. Only an
/// unrecorded fallback can become `Stopped` after an OS connection refusal.
enum StatusTarget {
    Explicit(String),
    Recorded(String),
    Configured {
        url: String,
        address: std::net::SocketAddrV4,
    },
}

impl StatusTarget {
    fn url(&self) -> &str {
        match self {
            Self::Explicit(url) | Self::Recorded(url) | Self::Configured { url, .. } => url,
        }
    }

    async fn observe(self, client: &BatchalignClient) -> ServerObservation {
        if let Self::Configured { address, .. } = &self {
            // HTTP transport errors may erase the OS cause or involve a proxy.
            // Only direct refusal of this known loopback socket proves absence.
            match tokio::time::timeout(
                std::time::Duration::from_secs(10),
                tokio::net::TcpStream::connect(*address),
            )
            .await
            {
                Ok(Ok(connection)) => drop(connection),
                Ok(Err(error)) => {
                    return match error.kind() {
                        std::io::ErrorKind::ConnectionRefused => ServerObservation::Stopped,
                        _ => ServerObservation::Refused {
                            message: format!(
                                "Cannot inspect the configured local service at {address}: {error}"
                            ),
                        },
                    };
                }
                Err(_) => {
                    return ServerObservation::Refused {
                        message: format!(
                            "Timed out inspecting the configured local service at {address}. No replacement was started."
                        ),
                    };
                }
            }
        }
        match client.health_check(self.url()).await {
            Ok(health) if health.build_hash == crate::build_hash() => ServerObservation::Ready {
                url: self.url().to_owned(),
                build_hash: health.build_hash,
            },
            Ok(_) => ServerObservation::Refused {
                message: "The managed server and installed CLI are different builds. Update/restart the managed service before processing; the desktop will not replace it.".to_owned(),
            },
            Err(error) => ServerObservation::Refused {
                message: format!("Cannot reach the discovered server at {}: {error}", self.url()),
            },
        }
    }
}

fn print_observation(observation: &ServerObservation) -> Result<(), CliError> {
    println!("{}", serde_json::to_string(observation)?);
    Ok(())
}

/// `serve sign-request`: the same-user header value for one request to this
/// account's own server, for the desktop app, which never holds the key.
/// The body is read from stdin; only the header value is printed.
pub async fn sign_request(args: &crate::cli::args::ServeSignRequestArgs) -> Result<(), CliError> {
    use std::io::Read;
    let layout = RuntimeLayout::from_env();
    let Some(own) = crate::same_user_key::OwnServer::read(layout.state_dir(), HandshakeSlot::Main)
        .map_err(|error| CliError::InvalidArgument(error.to_string()))?
    else {
        return Err(CliError::InvalidArgument(
            "no server of this account has published a same-user key".into(),
        ));
    };
    let mut body = Vec::new();
    std::io::stdin()
        .read_to_end(&mut body)
        .map_err(CliError::Io)?;
    let client = BatchalignClient::new()?;
    let header = client
        .sign_for_own(&own, &args.method.to_ascii_uppercase(), &args.path, &body)
        .await?;
    println!("{header}");
    Ok(())
}

/// `serve status`: check server health.
pub async fn status(args: &ServeStatusArgs) -> Result<(), CliError> {
    let client = BatchalignClient::new()?;
    let layout = RuntimeLayout::from_env();
    // `serve status` is a diagnostic command, so surfacing a bad config is part
    // of its job. The port is only a FALLBACK: this command wants to reach the
    // server actually running, which the published handshake names.
    let cfg = config::load_config_from_layout(&layout, None)?;

    let server = if let Some(ref s) = args.server {
        StatusTarget::Explicit(s.trim_end_matches('/').to_string())
    } else {
        // Try local daemon first
        // The daemon's port comes from the handshake it published, not from
        // `daemon.json`, which no longer mirrors it.
        let recorded = match daemon::read_daemon_info() {
            Ok(recorded) => recorded,
            // A diagnostic command: say so, then look for a server the other
            // ways rather than stopping here.
            Err(unreadable) => {
                if matches!(args.format, ServeStatusFormat::Json) {
                    return print_observation(&ServerObservation::Refused {
                        message: unreadable.to_string(),
                    });
                }
                eprintln!("warning: {unreadable}");
                None
            }
        };
        let published = match ServerHandshake::read(layout.state_dir(), HandshakeSlot::Main) {
            Ok(handshake) => handshake,
            Err(error) if matches!(args.format, ServeStatusFormat::Json) => {
                return print_observation(&ServerObservation::Refused {
                    message: error.to_string(),
                });
            }
            Err(error) => {
                eprintln!("warning: {error}");
                None
            }
        };
        // Bind target classification to the records just read; do not re-read
        // a handshake that another process may replace between these steps.
        let port = published
            .and_then(ServerHandshake::bound_port)
            .map(|port| port.get())
            .or_else(|| cfg.port.fixed().map(|port| port.get()));
        let url = port.map(|port| format!("http://127.0.0.1:{port}"));
        match (url, recorded, published, cfg.port.fixed()) {
            (Some(url), None, None, Some(port)) => StatusTarget::Configured {
                url,
                address: std::net::SocketAddrV4::new(std::net::Ipv4Addr::LOCALHOST, port.get()),
            },
            (Some(url), _, _, _) => StatusTarget::Recorded(url),
            (None, None, None, _) => {
                if matches!(args.format, ServeStatusFormat::Json) {
                    return print_observation(&ServerObservation::Stopped);
                }
                eprintln!("No local server discoverable. Pass --server URL.");
                return Ok(());
            }
            (None, _, _, _) => {
                if matches!(args.format, ServeStatusFormat::Json) {
                    return print_observation(&ServerObservation::Refused {
                        message: "A local service is recorded but has not published a listening port. Check its startup log before retrying.".to_owned(),
                    });
                }
                eprintln!("The recorded local service has not published a listening port.");
                return Ok(());
            }
        }
    };

    if matches!(args.format, ServeStatusFormat::Json) {
        return print_observation(&server.observe(&client).await);
    }
    match client.health_check(server.url()).await {
        Ok(health) => {
            eprintln!();
            eprintln!("Batchalign Server Status");
            eprintln!("{}", "-".repeat(40));
            eprintln!("URL:              {}", server.url());
            eprintln!("Status:           {}", health.status);
            eprintln!("Version:          {}", health.version);
            if !health.build_hash.is_empty() {
                eprintln!("Build:            {}", health.build_hash);
            }
            eprintln!("Workers free:     {}", health.workers_available);
            eprintln!("Active jobs:      {}", health.active_jobs);
            if !health.media_roots.is_empty() {
                eprintln!("Media:            {}", health.media_roots.join(", "));
            }
            eprintln!();
        }
        Err(e) => {
            eprintln!("error: cannot reach server at {}: {e}", server.url());
        }
    }

    Ok(())
}
