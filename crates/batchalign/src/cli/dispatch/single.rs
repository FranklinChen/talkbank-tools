//! Single-server dispatch: submit files to one server, poll, write results.

use crate::ReleasedCommand;
use crate::api::LanguageSpec;
use crate::cli::output::Overwrite;
use crate::command_model::{CommandIoProfile, command_spec};
use crate::options::CommandOptions;

use crate::cli::client::{BatchalignClient, server_label};
use crate::cli::error::CliError;
use crate::cli::progress::{BatchProgress, ProgressDisplay};
use crate::cli::tui::TuiProgress;

/// How the client transfers job inputs to one selected server.
///
/// Decided once, when the origin is parsed, and carried with the target: an
/// explicit loopback URL shares this machine's filesystem and gets the same
/// lossless path transport as an auto-discovered local daemon; any other
/// origin receives the inputs as content.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ServerTransport {
    /// A loopback server reads the inputs by path and writes beside them.
    SharedFilesystem,
    /// Transcript text over HTTP; the server needs nothing else.
    Content,
    /// Transcript text over HTTP, and the server must find each transcript's
    /// recording itself, in its own media mappings and roots. The operator is
    /// told so at submission, because a recording that exists only on this
    /// machine will not be found there.
    ContentResolvingAudio,
}

/// One validated explicit server paired with a transport that can carry the
/// selected command's inputs.
///
/// Construction refuses the one impossible state, "remote server plus
/// recordings as inputs": no transport carries a recording to another host.
/// It is keyed on what the command's inputs ARE, never on whether the command
/// touches audio: `align` touches audio and still travels as a transcript,
/// because the execution host resolves the recording itself. Once BA3 gains
/// media-body upload, `MediaInput` gets a new transport variant rather than a
/// weaker check here.
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
            // `to_canonical` first: an IPv4-mapped IPv6 address
            // (`::ffff:127.0.0.1`) is IPv4 loopback, but `Ipv6Addr::is_loopback`
            // recognizes only `::1`, so the mapped form read as remote.
            unbracketed.eq_ignore_ascii_case("localhost")
                || unbracketed
                    .parse::<std::net::IpAddr>()
                    .is_ok_and(|address| address.to_canonical().is_loopback())
        }))
    }

    /// Pair an operator-selected origin with the only transport it can use.
    pub(super) fn parse_explicit(url: &str, command: ReleasedCommand) -> Result<Self, CliError> {
        let loopback = Self::parse_origin(url)?;
        let transport = if loopback {
            ServerTransport::SharedFilesystem
        } else {
            match command_spec(command).io_profile {
                CommandIoProfile::Text => ServerTransport::Content,
                CommandIoProfile::ResolvedAudio => ServerTransport::ContentResolvingAudio,
                CommandIoProfile::MediaInput => {
                    return Err(CliError::InvalidArgument(format!(
                        "cannot send local audio to non-loopback server {url}: `{command}` takes \
                         recordings as its inputs, and no transport carries a recording to \
                         another host. Run it there, or give --server a loopback origin; see \
                         the Server Mode page."
                    )));
                }
            }
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
            transport: ServerTransport::SharedFilesystem,
        })
    }

    pub(super) fn url(&self) -> &str {
        &self.url
    }

    /// How a submission to this target travels.
    pub(super) fn transport(&self) -> ServerTransport {
        self.transport
    }
}

/// Map a `TuiCancelSignal` from the rendering thread into a wire-format
/// `CancellationRequest`. Captures source=Tui, the host machine name and
/// caller PID, plus the in-flight filename the TUI snapshot recorded at
/// the moment the user confirmed cancel ('y' after 'c').
///
/// The host is absent (`None`) when the OS does not report a name,
/// rather than recorded as a made-up `"unknown"` machine.
fn build_tui_cancel_provenance(
    signal: crate::cli::tui::TuiCancelSignal,
) -> crate::api::CancellationRequest {
    use crate::api::{
        CallerHost, CallerPid, CancelReason, CancelSource, CancellationRequest, DisplayPath,
    };

    let host = sysinfo::System::host_name().ok().map(CallerHost::from);

    CancellationRequest {
        source: Some(CancelSource::Tui),
        host,
        pid: Some(CallerPid(std::process::id())),
        reason: Some(CancelReason::from("user-pressed-cancel")),
        correlation_id: None,
        in_flight_filename: signal.in_flight_filename.map(DisplayPath::from),
    }
}

use super::display::{DashboardLaunch, Session, open_dashboard};
use super::helpers::poll_and_write_incrementally;
use super::paths::{prepare_content_submission, prepare_paths_submission};
use super::{refuse_foreign_server_build, server_supports_command};
use crate::cli::args::InputKind;
use crate::cli::args::Progress;
use crate::cli::discover::PassthroughMode;

/// Submit files to a single server, poll for completion, write results.
///
/// The command is the one `options` names; see `prepare_paths_submission`.
#[allow(clippy::too_many_arguments)]
pub(super) async fn dispatch_single_server(
    client: &BatchalignClient,
    target: &ServerTarget,
    lang: &LanguageSpec,
    num_speakers: u32,
    input_kind: InputKind,
    inputs: &[std::path::PathBuf],
    out_dir: Option<&std::path::Path>,
    options: &CommandOptions,
    lexicon: Option<&str>,
    before: Option<&std::path::Path>,
    progress: Progress,
    session: Session,
    launch: DashboardLaunch,
    overwrite: Option<Overwrite>,
) -> Result<(), CliError> {
    let command = options.command();
    let server_url = target.url();
    // Health check
    let health = match client.health_check(server_url).await {
        Ok(h) => h,
        Err(e) => {
            return Err(e);
        }
    };
    // Before anything is submitted: a server of another build would produce
    // results this CLI then writes as its own.
    refuse_foreign_server_build(server_url, &health)?;

    // Check capabilities
    if !server_supports_command(&health.capabilities, command) {
        return Err(CliError::UnsupportedCommand {
            server: server_label(server_url).to_string(),
            command,
        });
    }

    // The transport was decided when the origin was parsed and alone decides
    // path versus content submission. Producer selection was already resolved
    // by dispatch.
    let (submission, destinations) = match target.transport {
        ServerTransport::SharedFilesystem => {
            let Some(prepared) = prepare_paths_submission(
                options,
                lang,
                num_speakers,
                input_kind,
                inputs,
                out_dir,
                lexicon,
                before,
                &health.media_mapping_keys,
                overwrite,
                PassthroughMode::Copy,
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

            (prepared.submission, prepared.destinations)
        }
        ServerTransport::Content | ServerTransport::ContentResolvingAudio => {
            let Some(prepared) = prepare_content_submission(
                options,
                lang,
                num_speakers,
                input_kind,
                inputs,
                out_dir,
                lexicon,
                overwrite,
                PassthroughMode::Copy,
            )?
            else {
                eprintln!("warning: no files found for {input_kind:?} input");
                return Ok(());
            };
            crate::cli::discover::print_passthrough_skips(&prepared.passthrough);

            eprintln!("Found {} file(s) to submit.\n", prepared.total_files);
            eprintln!("Submitting to {server_url}...");
            if let ServerTransport::ContentResolvingAudio = target.transport {
                eprintln!(
                    "note: the server finds each transcript's recording in its own media \
                     mappings and roots; a recording that exists only on this machine will \
                     not be found there.\n"
                );
            }

            (prepared.submission, prepared.destinations)
        }
    };

    let info = client.submit_job(server_url, &submission).await?;
    let job_id = &info.job_id;
    let total_files = info.total_files;
    eprintln!("Job {job_id} submitted ({total_files} file(s))");

    let dashboard_url = format!("{server_url}/dashboard/jobs/{job_id}");
    eprintln!("Dashboard: {dashboard_url}\n");

    open_dashboard(&dashboard_url, launch);

    // Poll and write incrementally
    if !info.status.is_terminal() {
        // The TUI needs a terminal to draw on; a plain session gets bars.
        if matches!((progress, session), (Progress::Tui, Session::Interactive)) {
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
            let mut tui_handle = crate::blocking::spawn_in_span(move || {
                crate::cli::tui::run_tui_loop(tui_runtime, Some(cancel_tx))
            });

            // Poll on current task, pinned so it survives TUI exit
            let poll_fut = poll_and_write_incrementally(
                client,
                server_url,
                job_id,
                total_files as u64,
                &destinations,
                command.as_wire_name(),
                &tui_progress,
            );
            tokio::pin!(poll_fut);

            tokio::select! {
                result = &mut poll_fut => {
                    // Polling ended, with the job's outcome or an error
                    // (diagnosed output and failed jobs are errors here, so
                    // they set the exit code). Either way the TUI is told
                    // the run is over, which is idempotent for paths that
                    // already told it, and it is awaited before the outcome
                    // propagates: the caller exits the process on an error,
                    // which would skip the TUI thread's terminal restore and
                    // its final summary.
                    tui_progress.finish();
                    let _ = tui_handle.await;
                    result?;
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
                &destinations,
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
    use super::{ServerTarget, ServerTransport};
    use crate::ReleasedCommand;

    #[test]
    fn explicit_loopback_server_constructs_shared_filesystem_transport() {
        assert_eq!(
            ServerTarget::parse_explicit("http://127.0.0.1:8002", ReleasedCommand::Transcribe)
                .unwrap()
                .transport,
            ServerTransport::SharedFilesystem
        );
    }

    /// One command per `CommandIoProfile` variant; which commands hold which
    /// profile is the catalog pin table's job. `align` is the transcript-input
    /// command whose recording the server resolves itself, and its transport
    /// says so.
    #[test]
    fn explicit_remote_server_admits_transcript_input_commands_as_content() {
        for (command, transport) in [
            (ReleasedCommand::Morphotag, ServerTransport::Content),
            (
                ReleasedCommand::Align,
                ServerTransport::ContentResolvingAudio,
            ),
        ] {
            assert_eq!(
                ServerTarget::parse_explicit("http://media-host.example.org:8001", command)
                    .unwrap_or_else(|error| panic!("{command} must reach a remote server: {error}"))
                    .transport,
                transport,
                "{command}"
            );
        }
    }

    #[test]
    fn explicit_remote_server_refuses_recording_input_commands() {
        let error = ServerTarget::parse_explicit(
            "http://media-host.example.org:8001",
            ReleasedCommand::Transcribe,
        )
        .expect_err("transcribe must be refused for a remote server");
        assert!(
            error.to_string().contains("recordings"),
            "the refusal must say why: {error}"
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

    /// Every spelling of this machine's loopback is local, including the
    /// IPv4-mapped IPv6 form, which `Ipv6Addr::is_loopback` alone misses.
    #[test]
    fn every_loopback_spelling_is_local() {
        for origin in [
            "http://localhost:8002",
            "http://127.0.0.1:8002",
            "http://[::1]:8002",
            "http://[::ffff:127.0.0.1]:8002",
        ] {
            assert_eq!(
                ServerTarget::parse_explicit(origin, ReleasedCommand::Transcribe)
                    .unwrap_or_else(|error| panic!("{origin} is loopback: {error}"))
                    .transport,
                ServerTransport::SharedFilesystem,
                "{origin}"
            );
        }
    }

    #[test]
    fn shared_filesystem_target_refuses_a_remote_origin() {
        assert!(ServerTarget::parse_shared_filesystem("https://worker.example.org").is_err());
    }
}
