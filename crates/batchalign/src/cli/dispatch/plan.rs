//! `--dry-run`: the submission a run would make, planned and printed, never
//! sent.
//!
//! The plan is not a description written beside the run. It is built by the
//! same preparation the run hands its daemon (`prepare_paths_submission` or
//! `prepare_content_submission`, giving one [`PreparedSubmission`]), after
//! the same argument checks, so it cannot drift from what the run would do.
//! What differs is only what is withheld: no daemon is started, no pool is
//! opened, no provider is called and no job is submitted. A running server
//! may be asked for its health, which the run would also ask before
//! submitting, to learn where it would look for media.

use std::fmt;
use std::path::{Path, PathBuf};

use crate::ReleasedCommand;
use crate::api::{HealthResponse, LanguageSpec};
use crate::cli::args::InputKind;
use crate::cli::client::{self, BatchalignClient};
use crate::cli::daemon::{self, RequestedRuntime, RuntimeRefusal};
use crate::cli::discover::PassthroughMode;
use crate::cli::error::CliError;
use crate::cli::output::Overwrite;
use crate::config::{RuntimeLayout, ServerConfig, load_config_from_layout};
use crate::options::CommandOptions;

use super::paths::{PreparedSubmission, prepare_content_submission, prepare_paths_submission};
use super::single::{ServerTarget, ServerTransport};
use super::{
    DirectPace, LocalRoute, ServerRequest, health_label, probe_local_server,
    refuse_foreign_server_build, server_supports_command,
};

/// Whether this invocation submits its job or only plans and prints it.
///
/// The plan state carries the inputs the run would have refused as missing:
/// under `Submit` resolution stops at the first such input, so the list can
/// exist only here, and the type says so.
#[derive(Debug)]
pub enum Execution {
    /// Submit the job (the default).
    Submit,
    /// `--dry-run`: plan, print, exit 0.
    PlanOnly {
        /// The refusal the run would have raised for each named input that
        /// does not exist, in argument order.
        unresolved_inputs: Vec<CliError>,
    },
}

/// Everything `dispatch` knows when it decides to plan instead of run.
///
/// The command is not a field: the options name it (`options.command()`), so
/// the two cannot disagree. Where the job would run is one [`ServerRequest`],
/// the value the run itself routes on.
pub(super) struct PlanRequest<'a> {
    pub lang: &'a LanguageSpec,
    pub num_speakers: u32,
    pub input_kind: InputKind,
    pub server: ServerRequest<'a>,
    pub inputs: &'a [PathBuf],
    pub out_dir: Option<&'a Path>,
    pub options: &'a CommandOptions,
    pub lexicon: Option<&'a str>,
    pub before: Option<&'a Path>,
    pub runtime: RequestedRuntime,
    pub memory_tier: Option<crate::types::runtime::MemoryTierKind>,
    pub overwrite: Option<Overwrite>,
    pub unresolved_inputs: Vec<CliError>,
}

/// Where the run would execute, found without starting anything.
#[derive(Debug)]
enum ExecutionTarget {
    /// `--server URL`, answering, of this build, able to run the command.
    ExplicitServer {
        url: String,
        transport: ServerTransport,
        label: String,
    },
    /// A local daemon or server already answering on the configured port.
    RunningLocalServer { url: String, label: String },
    /// None is running; the run would start the local daemon from
    /// `server.yaml` (`auto_daemon: true`). Nothing was started.
    DaemonWouldStart,
    /// None is running and `auto_daemon` is off: the run would report "no
    /// server available" and do nothing.
    NoServer,
    /// `--no-server` or `--sequential`: this process runs the job itself.
    Direct(DirectPace),
}

impl ExecutionTarget {
    /// How the submission would travel: by path on a shared filesystem, or
    /// as content to another host.
    fn transport(&self) -> ServerTransport {
        match self {
            Self::ExplicitServer { transport, .. } => *transport,
            Self::RunningLocalServer { .. }
            | Self::DaemonWouldStart
            | Self::NoServer
            | Self::Direct(_) => ServerTransport::SharedFilesystem,
        }
    }
}

impl fmt::Display for ExecutionTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ExplicitServer {
                url,
                transport,
                label,
            } => {
                let how = match transport {
                    ServerTransport::SharedFilesystem => "loopback; inputs read by path",
                    ServerTransport::Content => "another host; transcripts sent as content",
                    ServerTransport::ContentResolvingAudio => {
                        "another host; transcripts sent as content, recordings resolved there"
                    }
                };
                write!(f, "explicit server {url} ({how}; {label})")
            }
            Self::RunningLocalServer { url, label } => {
                write!(f, "local server already running at {url} ({label})")
            }
            Self::DaemonWouldStart => f.write_str(
                "the local daemon, which the run would start (server.yaml auto_daemon: true); \
                 nothing was started",
            ),
            Self::NoServer => f.write_str(
                "no server: none is running and auto_daemon is off, so the run would report \
                 \"no server available\" and do nothing",
            ),
            Self::Direct(DirectPace::Sequential) => {
                f.write_str("this process, directly (--sequential: one file at a time)")
            }
            Self::Direct(DirectPace::Pooled) => f.write_str("this process, directly (--no-server)"),
        }
    }
}

/// The places the execution host searches for a transcript's recording, and
/// the media mapping keys it knows: from the server that answered, or from
/// the configuration the daemon would start with.
#[derive(Debug)]
struct MediaPlaces {
    mapping_keys: Vec<String>,
    roots: Vec<String>,
}

impl MediaPlaces {
    fn from_health(health: &HealthResponse) -> Self {
        Self {
            mapping_keys: health.media_mapping_keys.clone(),
            roots: health.media_roots.clone(),
        }
    }

    fn from_config(cfg: &ServerConfig) -> Self {
        Self {
            mapping_keys: cfg
                .media_mappings
                .keys()
                .map(|key| key.as_str().to_owned())
                .collect(),
            roots: cfg
                .media_roots
                .iter()
                .map(|root| root.as_ref().display().to_string())
                .collect(),
        }
    }
}

/// The plan: the prepared submission the run would send, where it would run
/// and where that host would look for media, and the inputs the run would
/// have refused.
pub(super) struct SubmissionPlan {
    command: ReleasedCommand,
    target: ExecutionTarget,
    places: MediaPlaces,
    out_dir: Option<PathBuf>,
    prepared: Option<PreparedSubmission>,
    unresolved_inputs: Vec<CliError>,
}

/// Build the plan. Nothing here submits, starts a daemon or calls a provider;
/// the only network it may use is a health check of a server that is already
/// running, which the run would also make before submitting.
pub(super) async fn build_plan(
    request: PlanRequest<'_>,
    layout: &RuntimeLayout,
) -> Result<SubmissionPlan, CliError> {
    let (target, places) = resolve_target(&request, layout).await?;
    let prepared = match target.transport() {
        ServerTransport::SharedFilesystem => prepare_paths_submission(
            request.options,
            request.lang,
            request.num_speakers,
            request.input_kind,
            request.inputs,
            request.out_dir,
            request.lexicon,
            request.before,
            &places.mapping_keys,
            request.overwrite,
            PassthroughMode::Plan,
        )?,
        ServerTransport::Content | ServerTransport::ContentResolvingAudio => {
            prepare_content_submission(
                request.options,
                request.lang,
                request.num_speakers,
                request.input_kind,
                request.inputs,
                request.out_dir,
                request.lexicon,
                request.overwrite,
                PassthroughMode::Plan,
            )?
        }
    };
    Ok(SubmissionPlan {
        command: request.options.command(),
        target,
        places,
        out_dir: request.out_dir.map(Path::to_path_buf),
        prepared,
        unresolved_inputs: request.unresolved_inputs,
    })
}

/// Where the run would execute, by the same routing `dispatch` uses, minus
/// every step that would start or submit something.
async fn resolve_target(
    request: &PlanRequest<'_>,
    layout: &RuntimeLayout,
) -> Result<(ExecutionTarget, MediaPlaces), CliError> {
    // 1. An explicit --server identifies the execution host, exactly as the
    // run reads it: one URL, the runtime flags refused beside it, the
    // transport decided from the origin and the command, the server's build
    // and capabilities checked before anything else.
    let command = request.options.command();
    let local = match request.server {
        ServerRequest::Local(local) => local,
        ServerRequest::Named(server) => {
            let urls = client::parse_servers(server);
            let [url] = urls.as_slice() else {
                return Err(CliError::InvalidArgument(format!(
                    "--server names {} URLs; a job runs on exactly one server",
                    urls.len()
                )));
            };
            RuntimeRefusal::admit_unrecorded(request.runtime, url)?;
            let target = ServerTarget::parse_explicit(url, command)?;
            let client = BatchalignClient::new()?;
            let health = client.health_check(url).await?;
            refuse_foreign_server_build(url, &health)?;
            if !server_supports_command(&health.capabilities, command) {
                return Err(CliError::UnsupportedCommand {
                    server: client::server_label(url).to_string(),
                    command,
                });
            }
            let places = MediaPlaces::from_health(&health);
            return Ok((
                ExecutionTarget::ExplicitServer {
                    url: url.clone(),
                    transport: target.transport(),
                    label: health_label(&health),
                },
                places,
            ));
        }
    };

    let mut cfg = load_config_from_layout(layout, None)?;
    if let Some(tier) = request.memory_tier {
        cfg.memory_tier = Some(tier);
    }
    match local {
        LocalRoute::Direct(pace) => {
            return Ok((
                ExecutionTarget::Direct(pace),
                MediaPlaces::from_config(&cfg),
            ));
        }
        LocalRoute::Managed => {}
    }

    // 2. A local server already answering on the configured port: the run
    // would reuse it. Asked for its health only, as the run would be.
    if let Some(url) = daemon::local_server_url(layout, cfg.port)
        && let Some(health) = probe_local_server(&url).await
    {
        let places = MediaPlaces::from_health(&health);
        return Ok((
            ExecutionTarget::RunningLocalServer {
                url,
                label: health_label(&health),
            },
            places,
        ));
    }

    // 3. None running: the run would start one (or, with auto_daemon off, do
    // nothing). The daemon it would start loads this same configuration, so
    // its media places are read from it.
    let target = if cfg.auto_daemon {
        ExecutionTarget::DaemonWouldStart
    } else {
        ExecutionTarget::NoServer
    };
    Ok((target, MediaPlaces::from_config(&cfg)))
}

impl fmt::Display for SubmissionPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "batchalign3 dry run: nothing was submitted.")?;
        writeln!(f)?;
        writeln!(f, "command:    {}", self.command)?;
        writeln!(f, "runs on:    {}", self.target)?;
        match &self.out_dir {
            Some(dir) => writeln!(f, "output:     {}", dir.display())?,
            None => writeln!(f, "output:     beside each input")?,
        }
        if self.places.roots.is_empty() && self.places.mapping_keys.is_empty() {
            writeln!(
                f,
                "media:      a recording is found only beside its transcript (no media roots \
                 or mappings configured on the execution host)"
            )?;
        } else {
            writeln!(f, "media:      the execution host searches, in order:")?;
            for root in &self.places.roots {
                writeln!(f, "              root    {root}")?;
            }
            for key in &self.places.mapping_keys {
                writeln!(f, "              mapping {key}")?;
            }
        }
        match &self.prepared {
            None => {
                writeln!(f, "inputs:     none found; the run would submit nothing")?;
            }
            Some(prepared) => {
                let submission = &prepared.submission;
                writeln!(f, "language:   {}", submission.lang)?;
                writeln!(f, "speakers:   {}", submission.num_speakers.0)?;
                if !submission.media_mapping.as_str().is_empty() {
                    writeln!(
                        f,
                        "mapping:    {} / {}",
                        submission.media_mapping.as_str(),
                        submission.media_subdir.as_str()
                    )?;
                }
                match serde_json::to_string_pretty(&submission.options) {
                    Ok(json) => {
                        writeln!(f, "options:    (engines and models as submitted)")?;
                        for line in json.lines() {
                            writeln!(f, "            {line}")?;
                        }
                    }
                    Err(error) => writeln!(f, "options:    not serializable: {error}")?,
                }
                writeln!(f)?;
                writeln!(f, "inputs ({}):", prepared.total_files)?;
                let mut current_source: Option<&Path> = None;
                for destination in prepared.destinations.planned() {
                    if current_source != Some(destination.source) {
                        current_source = Some(destination.source);
                        writeln!(f, "  {}", destination.source.display())?;
                    }
                    writeln!(
                        f,
                        "    -> {}  [{}; {}]",
                        destination.path.display(),
                        destination.content_type,
                        destination.decision
                    )?;
                }
                let would_copy = prepared.passthrough.copied().count();
                let withheld = prepared.passthrough.skipped().count();
                if would_copy > 0 || withheld > 0 {
                    writeln!(
                        f,
                        "passthrough: {would_copy} non-input file(s) would be copied to the \
                         output directory, {withheld} withheld"
                    )?;
                }
            }
        }
        if !self.unresolved_inputs.is_empty() {
            writeln!(f)?;
            writeln!(
                f,
                "unresolved inputs ({}): the run would refuse these before submitting",
                self.unresolved_inputs.len()
            )?;
            for refusal in &self.unresolved_inputs {
                writeln!(f, "  {refusal}")?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::args::Progress;
    use crate::cli::dispatch::{DispatchRequest, dispatch};

    /// A synthetic recordings directory and an isolated state directory with
    /// auto_daemon off, so no route can start the test binary as a daemon.
    fn fixture() -> (tempfile::TempDir, PathBuf, PathBuf) {
        let root = tempfile::tempdir().unwrap();
        let recordings = root.path().join("recordings");
        let state = root.path().join("state");
        std::fs::create_dir_all(&recordings).unwrap();
        std::fs::create_dir_all(&state).unwrap();
        std::fs::write(state.join("server.yaml"), "auto_daemon: false\n").unwrap();
        for name in ["one.wav", "two.mp3"] {
            std::fs::write(recordings.join(name), b"synthetic recording bytes").unwrap();
        }
        (root, recordings, state)
    }

    fn english() -> LanguageSpec {
        LanguageSpec::Resolved(crate::api::LanguageCode3::eng())
    }

    fn transcribe() -> CommandOptions {
        crate::recipe_runner::runtime::test_options(ReleasedCommand::Transcribe)
    }

    fn plan_request<'a>(
        lang: &'a LanguageSpec,
        options: &'a CommandOptions,
        inputs: &'a [PathBuf],
        unresolved_inputs: Vec<CliError>,
    ) -> PlanRequest<'a> {
        PlanRequest {
            lang,
            num_speakers: 2,
            input_kind: InputKind::Media,
            server: ServerRequest::Local(LocalRoute::Direct(DirectPace::Pooled)),
            inputs,
            out_dir: None,
            options,
            lexicon: None,
            before: None,
            runtime: RequestedRuntime::NONE,
            memory_tier: None,
            overwrite: None,
            unresolved_inputs,
        }
    }

    fn dispatch_request<'a>(
        inputs: &'a [PathBuf],
        unresolved_inputs: Vec<CliError>,
    ) -> DispatchRequest<'a> {
        DispatchRequest {
            lang: english(),
            num_speakers: 2,
            input_kind: InputKind::Media,
            server_arg: None,
            inputs,
            out_dir: None,
            options: transcribe(),
            bank: None,
            subdir: None,
            lexicon: None,
            progress: Progress::Plain,
            dashboard: crate::cli::args::Dashboard::Closed,
            force_cpu: false,
            allow_mps: false,
            no_server: true,
            before: None,
            runtime: RequestedRuntime::NONE,
            sequential: false,
            memory_tier: None,
            overwrite: None,
            execution: Execution::PlanOnly { unresolved_inputs },
        }
    }

    /// Only the file the fixture wrote: nothing started, nothing recorded.
    fn state_dir_untouched(state: &Path) -> bool {
        let names: Vec<String> = std::fs::read_dir(state)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names == ["server.yaml"]
    }

    /// A dry run over a directory of recordings names each recording and the
    /// `.cha` it would write beside it, as a fresh file, and submits nothing:
    /// no job store, no daemon record, no output file.
    #[tokio::test]
    async fn a_dry_run_plans_each_recording_and_submits_nothing() {
        let (_root, recordings, state) = fixture();
        let inputs = vec![recordings.clone()];
        let layout = RuntimeLayout::from_state_dir(state.clone());
        let (lang, options) = (english(), transcribe());
        let plan = build_plan(plan_request(&lang, &options, &inputs, Vec::new()), &layout)
            .await
            .expect("a plan");
        let text = plan.to_string();
        let canonical = recordings.canonicalize().unwrap();
        for (media, cha) in [("one.wav", "one.cha"), ("two.mp3", "two.cha")] {
            assert!(
                text.contains(&canonical.join(media).display().to_string()),
                "{text}"
            );
            assert!(
                text.contains(&format!(
                    "-> {}  [chat; new file]",
                    canonical.join(cha).display()
                )),
                "{text}"
            );
            assert!(
                !recordings.join(cha).exists(),
                "the plan must write no output"
            );
        }
        assert!(text.contains("nothing was submitted"), "{text}");
        assert!(
            text.contains("this process, directly (--no-server)"),
            "{text}"
        );
        assert!(text.contains("inputs (2):"), "{text}");
        assert!(state_dir_untouched(&state));

        // The same through `dispatch`, which is what the CLI calls.
        dispatch(dispatch_request(&inputs, Vec::new()), &layout)
            .await
            .expect("a dry run exits cleanly");
        assert!(state_dir_untouched(&state));
        assert!(!recordings.join("one.cha").exists());
    }

    /// A missing recording is reported in the plan as the refusal the run
    /// would have raised, and the dry run still succeeds over the rest.
    #[tokio::test]
    async fn a_dry_run_reports_a_missing_recording_and_still_plans() {
        let (_root, recordings, state) = fixture();
        let inputs = vec![recordings.clone()];
        let layout = RuntimeLayout::from_state_dir(state.clone());
        let missing = recordings.join("absent.wav");
        let (lang, options) = (english(), transcribe());
        let plan = build_plan(
            plan_request(
                &lang,
                &options,
                &inputs,
                vec![CliError::InputMissing(missing.clone())],
            ),
            &layout,
        )
        .await
        .expect("a plan");
        let text = plan.to_string();
        assert!(text.contains("unresolved inputs (1)"), "{text}");
        assert!(
            text.contains(&format!("input path does not exist: {}", missing.display())),
            "{text}"
        );
        assert!(text.contains("inputs (2):"), "{text}");
        dispatch(
            dispatch_request(&inputs, vec![CliError::InputMissing(missing)]),
            &layout,
        )
        .await
        .expect("a dry run with a missing input still exits cleanly");
        assert!(state_dir_untouched(&state));
    }

    /// With no running server and auto_daemon on, the plan says the daemon
    /// would be started, and starts nothing.
    #[tokio::test]
    async fn a_dry_run_names_the_daemon_it_would_start_without_starting_it() {
        let (_root, recordings, state) = fixture();
        std::fs::write(state.join("server.yaml"), "auto_daemon: true\n").unwrap();
        let inputs = vec![recordings];
        let layout = RuntimeLayout::from_state_dir(state.clone());
        let (lang, options) = (english(), transcribe());
        let mut request = plan_request(&lang, &options, &inputs, Vec::new());
        request.server = ServerRequest::Local(LocalRoute::Managed);
        let plan = build_plan(request, &layout).await.expect("a plan");
        let text = plan.to_string();
        assert!(
            text.contains("the local daemon, which the run would start"),
            "{text}"
        );
        assert!(state_dir_untouched(&state), "no daemon record may appear");
    }
}
