//! `batchalign3 worker`: manage persistent worker daemons.
//!
//! Workers are started as foreground processes that listen on TCP localhost.
//! The OS service manager (launchd on macOS, systemd on Linux) is responsible
//! for backgrounding and auto-restart. The CLI provides `start`, `list`, and
//! `stop` subcommands for convenience.

use crate::worker::python::resolve_python_executable;
use crate::worker::registry::{self, RegistryEntry};
use crate::worker::tcp_handle::{TcpWorkerHandle, TcpWorkerInfo};
use crate::worker::{WorkerPid, WorkerProfile};

use crate::cli::args::{WorkerAction, WorkerArgs, WorkerStartArgs, WorkerStopArgs};
use crate::cli::error::CliError;

/// Dispatch the `batchalign3 worker` subcommand.
pub async fn run(
    args: &WorkerArgs,
    verbose: u8,
    engine_overrides: Option<&crate::types::engines::EngineOverrides>,
) -> Result<(), CliError> {
    match &args.action {
        WorkerAction::Start(start_args) => start(start_args, verbose, engine_overrides),
        WorkerAction::List => list().await,
        WorkerAction::Stop(stop_args) => stop(stop_args).await,
    }
}

/// Start a worker as a foreground daemon process.
///
/// Execs `python -m batchalign.worker --transport tcp --profile ... --port ...`.
/// On Unix this replaces the current process; on Windows it spawns and waits.
fn start(
    args: &WorkerStartArgs,
    verbose: u8,
    engine_overrides: Option<&crate::types::engines::EngineOverrides>,
) -> Result<(), CliError> {
    let python_path = resolve_python_executable();

    let Some(profile) = WorkerProfile::try_from_name(&args.profile) else {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "Unknown worker profile: {}. Must be one of: gpu, stanza, io",
                args.profile
            ),
        )
        .into());
    };

    let mut cmd = crate::worker::python::worker_command(&python_path);
    // The daemon records this build in its registry entry; a server of a
    // different build refuses to adopt it rather than trust its engine
    // identities and wire contract.
    cmd.env(
        crate::worker::registry::BUILD_IDENTITY_ENV,
        crate::build_hash(),
    );
    cmd.arg("--transport")
        .arg("tcp")
        .arg("--profile")
        .arg(&args.profile)
        .arg("--lang")
        .arg(&args.lang)
        .arg("--host")
        .arg(&args.host);

    if args.port > 0 {
        cmd.arg("--port").arg(args.port.to_string());
    }

    if let Some(overrides) = engine_overrides {
        cmd.arg("--engine-overrides")
            .arg(overrides.to_dispatch_json_string());
    }

    if verbose > 0 {
        cmd.arg("--verbose").arg(verbose.to_string());
    }

    // Serve the way a server on this host would route to this daemon once it
    // adopts it: the same host-resolved device policy, the same decision.
    let (server_config, warnings) = crate::config::load_validated_config_from_layout(
        &crate::config::RuntimeLayout::from_env(),
        None,
    )?;
    for warning in warnings {
        eprintln!("warning: {warning}");
    }
    let effective = crate::host_facts::EffectiveConfig::resolve_from_server_config(&server_config);
    if effective.force_cpu {
        cmd.arg("--force-cpu");
    }
    let runtime = crate::worker::handle::WorkerRuntimeConfig {
        force_cpu: effective.force_cpu,
        gpu_thread_pool_size: effective.gpu_thread_pool_size,
        ..crate::worker::handle::WorkerRuntimeConfig::default()
    };
    cmd.args(crate::worker::serving::WorkerServing::decide(profile, &runtime).worker_args());

    eprintln!(
        "Starting {} worker (lang={}, host={}, port={})...",
        args.profile,
        args.lang,
        args.host,
        if args.port > 0 {
            args.port.to_string()
        } else {
            "auto".to_string()
        },
    );

    // Exec the Python process, this replaces the current process on Unix,
    // or spawns and waits on Windows.
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;

        let error = cmd.exec();
        // exec() only returns on error.
        Err(std::io::Error::other(format!("Failed to exec worker process: {error}")).into())
    }

    #[cfg(not(unix))]
    {
        let status = cmd.status()?;
        if !status.success() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Worker process exited with status: {status}"),
            )
            .into());
        }
        Ok(())
    }
}

/// List all registered workers with health status.
async fn list() -> Result<(), CliError> {
    let registry_path = registry::default_registry_path();
    let entries = registry::read_registry(&registry_path);

    if entries.is_empty() {
        eprintln!("No workers registered in {}", registry_path.display());
        return Ok(());
    }

    eprintln!("Workers registered in {}:\n", registry_path.display());
    eprintln!(
        "{:<8} {:<8} {:<20} {:<6} {:<10}",
        "PROFILE", "LANG", "ADDRESS", "PID", "STATUS"
    );
    eprintln!("{}", "-".repeat(60));

    for entry in &entries {
        let status = check_worker_health(entry).await;
        eprintln!(
            "{:<8} {:<8} {:<20} {:<6} {:<10}",
            entry.profile,
            entry.lang,
            format!("{}:{}", entry.host, entry.port),
            entry.pid,
            status,
        );
    }

    Ok(())
}

/// Check if a registered worker is healthy by connecting and sending a health check.
async fn check_worker_health(entry: &RegistryEntry) -> &'static str {
    let Some(profile) = entry.worker_profile() else {
        return "unknown-profile";
    };

    let lang = match crate::api::WorkerLanguage::parse_untrusted(&entry.lang) {
        Ok(lang) => lang,
        Err(_) => return "invalid-lang",
    };

    let info = TcpWorkerInfo {
        host: entry.host.clone(),
        port: entry.port,
        profile,
        lang,
        engine_overrides: entry.engine_overrides.clone(),
        pid: WorkerPid(entry.pid),
        audio_task_timeout_s: 0,
        analysis_task_timeout_s: 0,
        // CLI-side health/stop probes use TcpWorkerHandle (one request at a
        // time), not SharedGpuTcpWorker: the dispatch semaphore is unused.
        gpu_thread_pool_size: 1,
    };

    match TcpWorkerHandle::connect(info).await {
        Ok(mut handle) => match handle.health_check().await {
            Ok(_) => "ready",
            Err(_) => "unhealthy",
        },
        Err(_) => "unreachable",
    }
}

/// Stop one or all workers.
async fn stop(args: &WorkerStopArgs) -> Result<(), CliError> {
    let registry_path = registry::default_registry_path();
    let entries = registry::read_registry(&registry_path);

    if entries.is_empty() {
        eprintln!("No workers registered.");
        return Ok(());
    }

    let targets: Vec<&RegistryEntry> = if args.all {
        entries.iter().collect()
    } else if args.port > 0 {
        entries.iter().filter(|e| e.port == args.port).collect()
    } else if !args.profile.is_empty() || !args.lang.is_empty() {
        entries
            .iter()
            .filter(|e| {
                (args.profile.is_empty() || e.profile == args.profile)
                    && (args.lang.is_empty() || e.lang == args.lang)
            })
            .collect()
    } else {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "Specify --port, --profile/--lang, or --all to select workers to stop",
        )
        .into());
    };

    if targets.is_empty() {
        eprintln!("No matching workers found.");
        return Ok(());
    }

    for entry in targets {
        let profile = RegistryEntry::worker_profile(entry).unwrap_or(WorkerProfile::Stanza);
        let lang = match crate::api::WorkerLanguage::parse_untrusted(&entry.lang) {
            Ok(lang) => lang,
            Err(error) => {
                eprintln!(
                    "Invalid worker language in registry for {} worker ({}:{}, pid={}): {}",
                    entry.profile, entry.host, entry.port, entry.pid, error
                );
                continue;
            }
        };

        let info = TcpWorkerInfo {
            host: entry.host.clone(),
            port: entry.port,
            profile,
            lang,
            engine_overrides: entry.engine_overrides.clone(),
            pid: WorkerPid(entry.pid),
            audio_task_timeout_s: 0,
            analysis_task_timeout_s: 0,
            // Stop probes use TcpWorkerHandle (single-shot); the dispatch
            // semaphore is unused on this path.
            gpu_thread_pool_size: 1,
        };

        match TcpWorkerHandle::connect(info).await {
            Ok(mut handle) => {
                let _ = handle.shutdown().await;
                eprintln!(
                    "Sent shutdown to {} worker ({}:{}, pid={})",
                    entry.profile, entry.host, entry.port, entry.pid
                );
            }
            Err(_) => {
                eprintln!(
                    "Cannot reach {} worker ({}:{}, pid={}), removing stale entry",
                    entry.profile, entry.host, entry.port, entry.pid
                );
                let _ = registry::remove_stale_entry(&registry_path, entry.pid);
            }
        }
    }

    Ok(())
}
