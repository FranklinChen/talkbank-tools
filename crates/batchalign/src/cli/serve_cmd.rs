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
use crate::host_facts::EffectiveConfig;
use crate::host_memory::HostMemoryRuntimeConfig;
use crate::host_policy::HostExecutionPolicy;
use crate::server_handshake::{HandshakeSlot, ServerHandshake};
use crate::worker::handle::WorkerRuntimeConfig;
use crate::worker::pool::PoolConfig;

use crate::cli::args::{ServeStartArgs, ServeStatusArgs};
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
    if let Some(workers) = args.workers {
        // `--workers N` is always an explicit override.
        cfg.max_workers_per_job = Some(workers as u32);
    }
    if let Some(timeout) = args.timeout {
        cfg.audio_task_timeout_s = Some(timeout);
    }

    if cfg.media_roots.is_empty() && cfg.media_mappings.is_empty() {
        eprintln!(
            "warning: no media_roots or media_mappings configured. \
             Align/transcribe commands will fail unless CHAT files reference \
             accessible media paths."
        );
    }

    if args.foreground {
        let tier = cfg.resolved_memory_tier();
        let host_policy = HostExecutionPolicy::from_server_config(&cfg);
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

        // CLI `--force-cpu` is a presence-only switch; convert to
        // `Some(true)` so the host-facts pipeline treats it as an
        // explicit override. Absent CLI flag leaves the
        // `cfg.force_cpu` field at whatever server.yaml provides
        // (default `None`, which falls through to the
        // recommendation).
        if force_cpu {
            cfg.force_cpu = Some(true);
        }
        if allow_mps {
            cfg.allow_mps = Some(true);
        }
        // Resolve operator overrides against the live host-facts
        // snapshot. The runtime forms below remain concrete `u32` /
        // `usize` values because every downstream consumer (worker
        // spawn args, dispatch_semaphore permit counts,
        // TcpWorkerInfo) already expects a single value, not an
        // override+recommendation pair.
        let effective = EffectiveConfig::resolve_from_server_config(&cfg);
        let worker_runtime = WorkerRuntimeConfig {
            force_cpu: effective.force_cpu,
            allow_mps: effective.allow_mps,
            gpu_thread_pool_size: effective.gpu_thread_pool_size,
            host_memory: HostMemoryRuntimeConfig::from_server_config(&cfg),
            memory_tier: tier,
            bootstrap_mode: host_policy.bootstrap_mode,
            ..WorkerRuntimeConfig::default()
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
            ready_timeout_s: cfg.worker_ready_timeout_s,
            // `recommend_max_total_workers` clamps to `[2, 32]` so the
            // `as usize` cast is always well-defined.
            max_total_workers: effective.max_total_workers as usize,
            checkout_wait_timeout: None,
            task_timeouts: cfg.task_timeouts(),
            ensure_task_timeout: cfg.ensure_task_timeout_s,
            worker_registry_path: cfg.worker_registry_path.clone(),
            test_delay_ms: 0,
            // Production: live host CPU loadavg gate, not a test override.
            cpu_gate_threshold_override: None,
        };
        crate::serve_with_runtime(
            cfg,
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
        // Forward workers to the background server process.
        if let Some(workers) = args.workers {
            cmd.args(["--workers", &workers.to_string()]);
        }
        // Forward timeout to the background server process.
        if let Some(timeout) = args.timeout {
            cmd.args(["--timeout", &timeout.to_string()]);
        }

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

/// `serve stop`: stop the server and daemon.
pub async fn stop() -> Result<(), CliError> {
    let layout = RuntimeLayout::from_env();

    // Daemons first, then the manually started server. Each outcome is
    // reported as what it was; a record that cannot be read is named and left
    // in place, and makes the command fail once everything else was tried.
    let outcomes = [
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

/// `serve status`: check server health.
pub async fn status(args: &ServeStatusArgs) -> Result<(), CliError> {
    let client = BatchalignClient::new()?;
    let layout = RuntimeLayout::from_env();
    // `serve status` is a diagnostic command, so surfacing a bad config is part
    // of its job. The port is only a FALLBACK: this command wants to reach the
    // server actually running, which the published handshake names.
    let cfg = config::load_config_from_layout(&layout, None)?;

    let server = if let Some(ref s) = args.server {
        s.trim_end_matches('/').to_string()
    } else {
        // Try local daemon first
        // The daemon's port comes from the handshake it published, not from
        // `daemon.json`, which no longer mirrors it.
        let recorded = match daemon::read_daemon_info() {
            Ok(recorded) => recorded,
            // A diagnostic command: say so, then look for a server the other
            // ways rather than stopping here.
            Err(unreadable) => {
                eprintln!("warning: {unreadable}");
                None
            }
        };
        let daemon_url = recorded.and_then(|info| {
            ServerHandshake::published_port(layout.state_dir(), HandshakeSlot::Main)
                .map(|port| (info.pid, format!("http://127.0.0.1:{port}")))
        });
        match daemon_url {
            Some((pid, url)) if client.health_check(&url).await.is_ok() => {
                eprintln!("Using local daemon (PID {pid})");
                url
            }
            // Either no daemon record, or its published port does not answer.
            // Falling back to the same URL would just re-probe what failed.
            _ => match daemon::local_server_url(&layout, cfg.port) {
                Some(url) => url,
                None => {
                    eprintln!("No local server discoverable. Pass --server URL.");
                    return Ok(());
                }
            },
        }
    };

    match client.health_check(&server).await {
        Ok(health) => {
            eprintln!();
            eprintln!("Batchalign Server Status");
            eprintln!("{}", "-".repeat(40));
            eprintln!("URL:              {server}");
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
            eprintln!("error: cannot reach server at {server}: {e}");
        }
    }

    Ok(())
}
