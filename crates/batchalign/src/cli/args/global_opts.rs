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
    /// default to 1).
    ///
    /// Applies to new daemons (auto-spawned or `--no-server` direct
    /// runs). When reusing an existing daemon (the common server-mode
    /// path), the per-job parallelism was fixed at the daemon's startup
    /// and `--workers` here does NOT override it, pass `--no-server`
    /// or restart the daemon to change the parallelism. A note is
    /// printed when `--workers` is provided but reuse is happening.
    #[arg(long, global = true)]
    pub workers: Option<usize>,

    /// Inference timeout in seconds for audio tasks (ASR, FA, speaker).
    /// Increase for very long recordings (default: 1800 = 30 minutes).
    ///
    /// Applies to new daemons (auto-spawned or `--no-server` direct
    /// runs). When reusing an existing daemon (the common server-mode
    /// path), the per-task ceiling was fixed at the daemon's startup
    /// and `--timeout` here does NOT override it, pass `--no-server`
    /// or restart the daemon to change the ceiling. A note is printed
    /// when `--timeout` is provided but reuse is happening.
    #[arg(long, global = true)]
    pub timeout: Option<u64>,

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

    /// Use full-screen TUI dashboard instead of progress bars (default for
    /// interactive terminals). Pass --no-tui to use simple progress bars.
    #[arg(long, action = ArgAction::SetTrue, default_value_t = true, global = true)]
    pub tui: bool,

    /// Disable full-screen TUI; use simple progress bars instead.
    #[arg(long = "no-tui", action = ArgAction::SetTrue, global = true)]
    pub no_tui: bool,

    /// Auto-open the submitted job in the browser dashboard after submission.
    /// Pass --no-open-dashboard to disable.
    ///
    /// Currently only macOS launches a browser automatically; other platforms
    /// still print the dashboard URL for manual use.
    #[arg(long, action = ArgAction::SetTrue, default_value_t = true, global = true)]
    pub open_dashboard: bool,

    /// Disable browser auto-open for submitted dashboard job pages.
    #[arg(long = "no-open-dashboard", action = ArgAction::SetTrue, global = true)]
    pub no_open_dashboard: bool,

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
}

/// Mutually constrained media-evidence cache flags at the Clap boundary.
///
/// Keeping these inputs together prevents unrelated global-option growth and
/// gives later admission code one owner for their relationship. The wire
/// compatibility flags remain separate because existing scripts use them.
#[derive(Args, Debug, Clone, Default)]
pub struct MediaCacheOpts {
    /// Bypass the media analysis cache.
    #[arg(
        long,
        global = true,
        conflicts_with_all = ["require_media_cache", "override_media_cache_tasks"]
    )]
    pub override_media_cache: bool,

    /// Refuse cache-backed media inference unless reusable evidence already
    /// exists. At those cache-backed boundaries, a miss cannot trigger
    /// inference.
    #[arg(
        long,
        global = true,
        conflicts_with_all = ["override_media_cache", "override_media_cache_tasks"]
    )]
    pub require_media_cache: bool,

    /// Bypass cache only for specific tasks (comma-separated).
    /// Honored for audio tasks: `forced_alignment`, `utr_asr`.
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
    /// Whether to use the full-screen TUI (resolves --tui / --no-tui).
    pub fn use_tui(&self) -> bool {
        self.tui && !self.no_tui
    }

    /// Whether to auto-open the browser dashboard (resolves --open-dashboard / --no-open-dashboard).
    pub fn use_open_dashboard(&self) -> bool {
        self.open_dashboard && !self.no_open_dashboard
    }
}
