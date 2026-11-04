use clap::{ArgAction, Args};

use super::parse_engine_overrides_json;
use crate::options::EngineOverrides;

/// Global options that apply to every command.
#[derive(Args, Debug, Clone)]
pub struct GlobalOpts {
    /// Increase verbosity (-v, -vv, -vvv).
    #[arg(short, long, action = ArgAction::Count, global = true)]
    pub verbose: u8,

    /// Maximum concurrent files per job (default: auto-tune; GPU commands
    /// default to 1). A positive count; leave it out for auto-tune.
    ///
    /// A daemon fixes this when it starts. The flag is honoured by a daemon
    /// this invocation starts and by a `--no-server` run; a running daemon
    /// that was started with another value refuses the invocation (nothing
    /// is submitted), naming both values: stop it with `batchalign3 serve
    /// stop` and run the command again, or drop the flag.
    #[arg(long, global = true)]
    pub workers: Option<crate::api::WorkerCap>,

    /// Inference timeout in seconds for audio tasks (ASR, FA, speaker).
    /// Increase for very long recordings (default: 1800 = 30 minutes). A
    /// positive count; leave it out for the built-in ceiling.
    ///
    /// A daemon fixes this when it starts, and it is honoured or refused
    /// exactly as `--workers` is.
    #[arg(long, global = true)]
    pub timeout: Option<crate::api::PositiveSeconds>,

    /// Disable MPS/CUDA and force CPU-only models.
    #[arg(long, global = true)]
    pub force_cpu: bool,

    /// Allow engines to use the Apple GPU (MPS). Off by default: MPS
    /// failures are rare but can hard-stall the machine (Apple driver
    /// deadlock, Apr 2026). Model dtypes stay float32 on MPS, and the
    /// speaker (diarization) stage never uses MPS regardless.
    #[arg(long, global = true, conflicts_with = "force_cpu")]
    pub allow_mps: bool,

    /// Remote server URL (or set BATCHALIGN_SERVER env var).
    #[arg(long, env = "BATCHALIGN_SERVER", global = true)]
    pub server: Option<String>,

    /// Skip auto-detection of a local server. By default, batchalign3
    /// checks if a server is running locally and routes work through it
    /// for fleet benefits (warm models, distributed processing, crash
    /// recovery). Use --no-server to force direct in-process execution.
    #[arg(long = "no-server", global = true)]
    pub no_server: bool,

    /// Media-evidence cache controls admitted as one CLI boundary.
    #[command(flatten)]
    pub media_cache: MediaCacheOpts,

    /// `--tui` / `--no-tui`: the two spellings of one [`Progress`] choice.
    #[command(flatten)]
    pub progress: ProgressFlags,

    /// `--open-dashboard` / `--no-open-dashboard`: the two spellings of one
    /// [`Dashboard`] choice.
    #[command(flatten)]
    pub dashboard: DashboardFlags,

    /// Directory for pipeline debug artifacts (CHAT/JSON fixtures for
    /// offline replay). Also enables dashboard algorithm trace collection.
    /// Env fallback: BATCHALIGN_DEBUG_DIR.
    #[arg(long, env = "BATCHALIGN_DEBUG_DIR", value_name = "PATH", global = true)]
    pub debug_dir: Option<std::path::PathBuf>,

    /// Engine overrides as JSON (e.g. '{"asr": "tencent", "fa": "cantonese_fa"}').
    ///
    /// Parsed ONCE, here, by the function that owns the payload's grammar, so
    /// this field holds the typed overrides themselves. There is no string
    /// left for a later reader to parse a second time, and no validator
    /// standing beside the parser with its own copy of the rules to drift
    /// from them.
    #[arg(
        long,
        value_name = "JSON",
        value_parser = parse_engine_overrides_json,
        global = true
    )]
    pub engine_overrides: Option<EngineOverrides>,

    /// Process files sequentially with minimal infrastructure. One worker
    /// per task type, no memory gate, no server. Ideal for small jobs on
    /// laptops where predictability matters more than throughput.
    #[arg(long, global = true, conflicts_with = "server")]
    pub sequential: bool,

    /// Override the auto-detected memory tier (small, medium, large, fleet).
    /// Forces the worker bootstrap mode and memory budgets for that tier
    /// regardless of actual system RAM. Useful for testing constrained-memory
    /// behavior on large machines.
    #[arg(long, global = true, value_name = "TIER")]
    pub memory_tier: Option<crate::types::runtime::MemoryTierKind>,

    /// `--dry-run`: plan the job and stop.
    #[command(flatten)]
    pub dry_run: DryRunFlag,
}

/// `--dry-run` at the Clap boundary. The field is private: the flag is
/// readable only as the [`RequestedExecution`] it selects.
#[derive(Args, Debug, Clone, Default)]
pub struct DryRunFlag {
    /// Plan the job and stop. Discover the inputs, decide every output
    /// destination, resolve the engines and the media places, name the daemon
    /// or server that would run it, print the plan, and exit 0: nothing is
    /// submitted, no daemon is started and no provider is called. The plan is
    /// the very submission the run would have sent, not a description beside
    /// it. Only a processing command has a job to plan; on any other command
    /// the flag is refused.
    #[arg(long, action = ArgAction::SetTrue, global = true)]
    dry_run: bool,
}

/// Whether the invocation submits its job or only plans and prints it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestedExecution {
    /// Submit the job (the default).
    Submit,
    /// `--dry-run`: plan, print, exit 0.
    PlanOnly,
}

impl DryRunFlag {
    /// The one choice the flag makes.
    pub fn resolve(&self) -> RequestedExecution {
        match self.dry_run {
            true => RequestedExecution::PlanOnly,
            false => RequestedExecution::Submit,
        }
    }
}

/// Mutually constrained media-evidence cache flags at the Clap boundary.
///
/// Keeping these inputs together prevents unrelated global-option growth and
/// gives later admission code one owner for their relationship. The wire
/// compatibility flags remain separate because existing scripts use them.
#[derive(Args, Debug, Clone, Default)]
pub struct MediaCacheOpts {
    /// Bypass the media analysis cache. For `align` it also asks for a
    /// transcript whose `%wor` is already complete to be aligned again from
    /// its recording, which is then required; without it such a file is
    /// refreshed from its own `%wor` and the recording is not read.
    #[arg(
        long,
        global = true,
        conflicts_with_all = ["require_media_cache", "override_media_cache_tasks", "require_media_cache_tasks"]
    )]
    pub override_media_cache: bool,

    /// Refuse cache-backed media inference unless reusable evidence already
    /// exists. At those cache-backed boundaries, a miss cannot trigger
    /// inference.
    #[arg(
        long,
        global = true,
        conflicts_with_all = ["override_media_cache", "override_media_cache_tasks", "require_media_cache_tasks"]
    )]
    pub require_media_cache: bool,

    /// Require existing evidence only for listed cache-backed tasks. Other
    /// tasks retain their own policy; e.g. require utr_asr while refreshing FA.
    #[arg(long, global = true, value_name = "TASKS", value_delimiter = ',',
        conflicts_with_all = ["override_media_cache", "require_media_cache"])]
    pub require_media_cache_tasks: Vec<String>,

    /// Bypass cache only for specific tasks (comma-separated).
    /// Honored for audio tasks: `forced_alignment`, `utr_asr`. Naming
    /// `forced_alignment` also asks `align` to re-align a complete-`%wor`
    /// transcript from its recording, as `--override-media-cache` does.
    /// Batchalign3 does not cache text-NLP tasks, so
    /// `morphosyntax`/`utterance_segmentation`/`translation` are
    /// accepted but are no-ops.
    #[arg(
        long,
        value_name = "TASKS",
        global = true,
        value_delimiter = ',',
        conflicts_with_all = ["override_media_cache", "require_media_cache"]
    )]
    pub override_media_cache_tasks: Vec<String>,
}

impl GlobalOpts {
    /// The daemon settings this invocation asked for, built once from
    /// `--workers` and `--timeout`.
    pub fn runtime(&self) -> crate::cli::daemon::RequestedRuntime {
        crate::cli::daemon::RequestedRuntime {
            workers: self.workers,
            timeout: self.timeout,
        }
    }
}

/// How a server-backed job's progress is shown in the terminal.
///
/// One value from two spellings. `--tui` and `--no-tui` used to be two
/// `bool`s read through `tui && !no_tui`, with `--tui` defaulting to true, so
/// the positive flag could never change anything: only `--no-tui` acted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Progress {
    /// The full-screen TUI, when the session is interactive (the default).
    Tui,
    /// Simple progress bars.
    Plain,
}

/// Whether the submitted job's dashboard page opens in the browser.
///
/// Three states, because the default is neither of the two flags: with no
/// flag the page opens once, when this invocation started the daemon, so a
/// shell loop over ten directories opens one tab. `--open-dashboard` used to
/// default to true and be read as `open_dashboard && !no_open_dashboard`, so
/// it was a no-op and every submission opened a tab; `--no-open-dashboard`
/// alone acted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dashboard {
    /// `--open-dashboard`: open the page for this submission, in an
    /// interactive session, whether or not the daemon was already running.
    Open,
    /// `--no-open-dashboard`: never open a browser.
    Closed,
    /// Neither flag: open the page only when this invocation started the
    /// daemon.
    OnDaemonStart,
}

/// The `--tui` / `--no-tui` pair at the Clap boundary.
///
/// The fields are private: the pair is readable only as the one [`Progress`]
/// it resolves to, and clap refuses both flags together, so there is no
/// `(true, true)` to decide.
#[derive(Args, Debug, Clone, Default)]
pub struct ProgressFlags {
    /// Show the full-screen TUI for server-backed jobs (the default in an
    /// interactive terminal).
    #[arg(long, action = ArgAction::SetTrue, global = true, conflicts_with = "no_tui")]
    tui: bool,

    /// Show simple progress bars instead of the full-screen TUI.
    #[arg(long = "no-tui", action = ArgAction::SetTrue, global = true)]
    no_tui: bool,
}

impl ProgressFlags {
    /// The one choice the two spellings make. Neither flag is the TUI.
    pub fn resolve(&self) -> Progress {
        match (self.tui, self.no_tui) {
            (_, true) => Progress::Plain,
            (true, false) | (false, false) => Progress::Tui,
        }
    }
}

/// The `--open-dashboard` / `--no-open-dashboard` pair at the Clap boundary.
///
/// Private fields and a clap conflict, as [`ProgressFlags`].
#[derive(Args, Debug, Clone, Default)]
pub struct DashboardFlags {
    /// Open the submitted job's dashboard page in the browser, even when the
    /// daemon was already running. Without either flag the page opens once,
    /// when this invocation starts the daemon. Only macOS launches a browser;
    /// other platforms print the URL.
    #[arg(
        long,
        action = ArgAction::SetTrue,
        global = true,
        conflicts_with = "no_open_dashboard"
    )]
    open_dashboard: bool,

    /// Never open the browser for submitted dashboard job pages.
    #[arg(long = "no-open-dashboard", action = ArgAction::SetTrue, global = true)]
    no_open_dashboard: bool,
}

impl DashboardFlags {
    /// The one choice the two spellings make.
    pub fn resolve(&self) -> Dashboard {
        match (self.open_dashboard, self.no_open_dashboard) {
            (_, true) => Dashboard::Closed,
            (true, false) => Dashboard::Open,
            (false, false) => Dashboard::OnDaemonStart,
        }
    }
}
