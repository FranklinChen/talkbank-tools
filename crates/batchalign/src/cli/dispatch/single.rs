//! Single-server dispatch: submit files to one server, poll, write results.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::ReleasedCommand;
use crate::api::JobSubmission;
use crate::options::CommandOptions;
use crate::{released_command_supports_paths_mode, released_command_uses_local_audio};

use crate::cli::client::{BatchalignClient, server_label};
use crate::cli::discover::{build_server_names, copy_nonmatching, infer_base_dir};
use crate::cli::error::CliError;
use crate::cli::progress::BatchProgress;
use crate::cli::tui::TuiProgress;

/// How the client may transfer job inputs to one selected server.
///
/// The private representation prevents call sites from passing an unexplained
/// boolean. An explicit loopback URL is still an explicit producer choice, but
/// it shares this machine's filesystem and therefore gets the same efficient,
/// lossless path transport as an auto-discovered local daemon.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ServerTransport(ServerTransportKind);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ServerTransportKind {
    SharedFilesystem,
    Content,
}

impl ServerTransport {
    fn uses_paths_for(self, command: ReleasedCommand) -> bool {
        self.0 == ServerTransportKind::SharedFilesystem
            && released_command_supports_paths_mode(command)
    }
}

/// One validated explicit server paired with a transport that can carry the
/// selected command's inputs.
///
/// Construction refuses the currently impossible state "remote server plus
/// client-local audio". Once BA3 gains media-body upload, that operation gets
/// a new transport variant rather than weakening this proof.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ServerTarget {
    url: String,
    transport: ServerTransport,
}

impl ServerTarget {
    fn parse_origin(url: &str) -> Result<bool, CliError> {
        let parsed = reqwest::Url::parse(url)
            .map_err(|error| CliError::InvalidArgument(format!("invalid --server URL: {error}")))?;
        if !matches!(parsed.scheme(), "http" | "https")
            || parsed.host_str().is_none()
            || !parsed.username().is_empty()
            || parsed.password().is_some()
            || parsed.path() != "/"
            || parsed.query().is_some()
            || parsed.fragment().is_some()
        {
            return Err(CliError::InvalidArgument(
                "--server must be an HTTP(S) origin without credentials, path, query, or fragment"
                    .into(),
            ));
        }

        Ok(parsed.host_str().is_some_and(|host| {
            let unbracketed = host.trim_start_matches('[').trim_end_matches(']');
            unbracketed.eq_ignore_ascii_case("localhost")
                || unbracketed
                    .parse::<std::net::IpAddr>()
                    .is_ok_and(|address| address.is_loopback())
        }))
    }

    /// Pair an operator-selected origin with the only transport it can use.
    pub(super) fn parse_explicit(url: &str, command: ReleasedCommand) -> Result<Self, CliError> {
        let loopback = Self::parse_origin(url)?;
        let transport = if loopback {
            ServerTransport(ServerTransportKind::SharedFilesystem)
        } else if released_command_uses_local_audio(command) {
            return Err(CliError::InvalidArgument(format!(
                "cannot send local audio to non-loopback server {url}; remote media upload is not implemented"
            )));
        } else {
            ServerTransport(ServerTransportKind::Content)
        };
        Ok(Self {
            url: url.to_owned(),
            transport,
        })
    }

    /// Admit a daemon URL only after proving that it names this host.
    pub(super) fn parse_shared_filesystem(url: &str) -> Result<Self, CliError> {
        if !Self::parse_origin(url)? {
            return Err(CliError::InvalidArgument(format!(
                "shared-filesystem server must use a loopback origin, got {url}"
            )));
        }
        Ok(Self {
            url: url.to_owned(),
            transport: ServerTransport(ServerTransportKind::SharedFilesystem),
        })
    }

    pub(super) fn url(&self) -> &str {
        &self.url
    }

    fn uses_paths_for(&self, command: ReleasedCommand) -> bool {
        self.transport.uses_paths_for(command)
    }
}

/// Map a `TuiCancelSignal` from the rendering thread into a wire-format
/// `CancellationRequest`. Captures source=Tui, the host machine name and
/// caller PID, plus the in-flight filename the TUI snapshot recorded at
/// the moment the user confirmed cancel ('y' after 'c').
///
/// Hostname falls back to `"unknown"` when the OS does not report one
/// (rare; tests, containers without `/etc/hostname`).
fn build_tui_cancel_provenance(
    signal: crate::cli::tui::TuiCancelSignal,
) -> crate::api::CancellationRequest {
    use crate::api::{
        CallerHost, CallerPid, CancelReason, CancelSource, CancellationRequest, DisplayPath,
    };

    let host = sysinfo::System::host_name().unwrap_or_else(|| "unknown".to_string());

    CancellationRequest {
        source: Some(CancelSource::Tui),
        host: Some(CallerHost::from(host)),
        pid: Some(CallerPid(std::process::id())),
        reason: Some(CancelReason::from("user-pressed-cancel")),
        correlation_id: None,
        in_flight_filename: signal.in_flight_filename.map(DisplayPath::from),
    }
}

use super::helpers::{
    classify_files, filter_files_for_command, inject_lexicon, maybe_open_dashboard,
    order_files_for_command, poll_and_write_incrementally,
};
use super::paths::prepare_paths_submission;
use super::{server_supports_command, warn_stale_server};
use crate::cli::args::InputKind;

/// Submit files to a single server, poll for completion, write results.
#[allow(clippy::too_many_arguments)]
pub(super) async fn dispatch_single_server(
    client: &BatchalignClient,
    target: &ServerTarget,
    command: ReleasedCommand,
    lang: &str,
    num_speakers: u32,
    input_kind: InputKind,
    inputs: &[std::path::PathBuf],
    out_dir: Option<&std::path::Path>,
    options: Option<&CommandOptions>,
    lexicon: Option<&str>,
    before: Option<&std::path::Path>,
    use_tui: bool,
    open_dashboard: bool,
) -> Result<(), CliError> {
    let server_url = target.url();
    // Health check
    let health = match client.health_check(server_url).await {
        Ok(h) => h,
        Err(e) => {
            return Err(e);
        }
    };
    warn_stale_server(server_url, &health);

    // Check capabilities
    if !server_supports_command(&health.capabilities, command) {
        return Err(CliError::UnsupportedCommand {
            server: server_label(server_url).to_string(),
            command,
        });
    }

    // The transport state records whether the selected server shares this
    // filesystem. Command metadata then decides whether that command supports
    // path submissions. Producer selection was already resolved by dispatch.
    let use_paths_mode = target.uses_paths_for(command);

    let (submission, effective_out, result_map, paths_mode) = if use_paths_mode {
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
            &health.media_mapping_keys,
        )?
        else {
            eprintln!("warning: no files found for {input_kind:?} input");
            return Ok(());
        };

        eprintln!("Found {} file(s) to submit.\n", prepared.total_files);
        crate::cli::discover::print_passthrough_skips(&prepared.passthrough);
        eprintln!("Submitting shared-filesystem job to {server_url}...");
        eprintln!(
            "note: the server must be able to read these input paths. Successful outputs will also be copied back to this machine.\n"
        );

        (
            prepared.submission,
            prepared.effective_out,
            HashMap::new(),
            true,
        )
    } else {
        let (files, outputs) =
            crate::cli::discover::discover_server_inputs(inputs, out_dir, input_kind)?;
        let (files, outputs) = filter_files_for_command(command, files, outputs);
        let (files, outputs) = order_files_for_command(command, files, outputs)?;

        if let Some(od) = out_dir {
            let mut passthrough = crate::cli::discover::PassthroughReport::default();
            for inp in inputs {
                if Path::new(inp).is_dir() {
                    let dir_report =
                        copy_nonmatching(Path::new(inp), Path::new(od), input_kind, command)?;
                    passthrough.extend_from(dir_report);
                }
            }
            crate::cli::discover::print_passthrough_skips(&passthrough);
        }

        let base_dir = infer_base_dir(inputs)?;
        let (server_names, result_map) = build_server_names(&files, &outputs, inputs)?;
        let (file_payloads, media_file_names) = classify_files(&files, &server_names)?;
        if file_payloads.is_empty() && media_file_names.is_empty() {
            eprintln!("warning: no files found for {input_kind:?} input");
            return Ok(());
        }

        let total_count = file_payloads.len() + media_file_names.len();
        eprintln!("Found {total_count} file(s) to submit.\n");

        let mut opts = options.cloned().unwrap_or_else(|| {
            CommandOptions::Morphotag(crate::options::MorphotagOptions {
                common: Default::default(),

                ..Default::default()
            })
        });
        inject_lexicon(&mut opts, lexicon)?;
        let debug_traces = opts.common().debug_dir.is_some();

        let effective_out = out_dir
            .map(PathBuf::from)
            .unwrap_or_else(|| base_dir.clone());

        (
            JobSubmission {
                command,
                lang: crate::api::LanguageSpec::try_from(lang)
                    .map_err(|e| CliError::InvalidArgument(format!("invalid language: {e}")))?,
                num_speakers: num_speakers.into(),
                files: file_payloads,
                media_files: media_file_names,
                media_mapping: Default::default(),
                media_subdir: Default::default(),
                source_dir: base_dir.to_string_lossy().to_string().into(),
                options: opts,
                paths_mode: false,
                source_paths: vec![],
                output_paths: vec![],
                display_names: vec![],
                debug_traces,
                before_paths: vec![],
            },
            effective_out,
            result_map,
            false,
        )
    };

    if !paths_mode {
        eprintln!("Submitting to {server_url}...");
    }
    let info = client.submit_job(server_url, &submission).await?;
    let job_id = &info.job_id;
    let total_files = info.total_files;
    eprintln!("Job {job_id} submitted ({total_files} file(s))");

    let dashboard_url = format!("{server_url}/dashboard/jobs/{job_id}");
    eprintln!("Dashboard: {dashboard_url}\n");

    maybe_open_dashboard(&dashboard_url, open_dashboard);

    // Poll and write incrementally
    if !info.status.is_terminal() {
        if use_tui && std::io::IsTerminal::is_terminal(&std::io::stdout()) {
            let (tui_progress, tui_runtime) =
                TuiProgress::new(total_files as u64, command.as_wire_name());
            let (cancel_tx, cancel_rx) =
                tokio::sync::oneshot::channel::<crate::cli::tui::TuiCancelSignal>();

            // Cancel task: awaits signal from TUI, posts cancel with full
            // caller provenance (source=Tui + hostname + PID + the in-flight
            // filename the TUI captured at confirm time). Server persists
            // these to the `cancellations` audit table so we can attribute
            // every cancel to a specific user gesture.
            let cc = client.clone();
            let cu = server_url.to_string();
            let cj = job_id.clone();
            tokio::spawn(async move {
                if let Ok(signal) = cancel_rx.await {
                    let provenance = build_tui_cancel_provenance(signal);
                    let _ = cc.cancel_job(&cu, &cj, provenance).await;
                }
            });

            // TUI on blocking thread
            let mut tui_handle = tokio::task::spawn_blocking(move || {
                crate::cli::tui::run_tui_loop(tui_runtime, Some(cancel_tx))
            });

            // Poll on current task, pinned so it survives TUI exit
            let poll_fut = poll_and_write_incrementally(
                client,
                server_url,
                job_id,
                total_files as u64,
                &result_map,
                &effective_out,
                command.as_wire_name(),
                &tui_progress,
            );
            tokio::pin!(poll_fut);

            tokio::select! {
                result = &mut poll_fut => {
                    result?;
                    // Job finished: wait for TUI to exit
                    let _ = tui_handle.await;
                }
                _ = &mut tui_handle => {
                    // User closed TUI: continue writing results to disk
                    eprintln!("\nDashboard closed: still writing results...");
                    poll_fut.await?;
                }
            }
        } else {
            let progress = BatchProgress::new(total_files as u64, command.as_wire_name());
            poll_and_write_incrementally(
                client,
                server_url,
                job_id,
                total_files as u64,
                &result_map,
                &effective_out,
                command.as_wire_name(),
                &progress,
            )
            .await?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{ServerTarget, ServerTransportKind};
    use crate::ReleasedCommand;

    #[test]
    fn explicit_loopback_server_constructs_shared_filesystem_transport() {
        assert_eq!(
            ServerTarget::parse_explicit("http://127.0.0.1:8002", ReleasedCommand::Transcribe)
                .unwrap()
                .transport
                .0,
            ServerTransportKind::SharedFilesystem
        );
    }

    #[test]
    fn explicit_remote_text_server_constructs_content_transport() {
        assert_eq!(
            ServerTarget::parse_explicit("https://worker.example.org", ReleasedCommand::Morphotag)
                .unwrap()
                .transport
                .0,
            ServerTransportKind::Content
        );
    }

    #[test]
    fn explicit_server_url_with_userinfo_is_refused_not_misclassified() {
        assert!(
            ServerTarget::parse_explicit(
                "http://localhost:8001@worker.example.org",
                ReleasedCommand::Morphotag
            )
            .is_err()
        );
    }

    #[test]
    fn shared_filesystem_target_refuses_a_remote_origin() {
        assert!(ServerTarget::parse_shared_filesystem("https://worker.example.org").is_err());
    }
}
