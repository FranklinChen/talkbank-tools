//! Terminal progress display for batch job polling.
//!
//! While the CLI polls a running server job, the user needs visual feedback
//! showing how many files have completed, which file is currently being
//! processed, and roughly how long the job has been running.
//!
//! This module provides [`BatchProgress`], an indicatif-based implementation
//! that renders a two-line display: a determinate progress bar for overall
//! file completion and a spinner showing the active processing stage. Both
//! implement the [`ProgressDisplay`] trait so the polling loop can be decoupled
//! from the rendering backend. The ratatui TUI uses a separate
//! reducer-message sender that implements the same trait while the render loop
//! owns UI state locally.

use crate::api::{FileStatusEntry, FileStatusKind, HealthResponse};
use indicatif::{ProgressBar, ProgressStyle};

/// Trait for receiving progress updates during job polling.
///
/// Implemented by `BatchProgress` (indicatif bars) and `TuiProgress`
/// (reducer-message sender for the ratatui runtime).
///
/// **Named `ProgressDisplay`, not `ProgressSink`, since 2026-07-29.** This is a
/// terminal RENDERER: it draws what the CLI has already learned by polling.
/// `ProgressSink` is reserved for a different concept from the upstream fork
/// whose mechanisms this crate is adopting: the seam a BACKEND pushes progress
/// into, introduced alongside `Dispatcher`. Where a name collides, the upstream
/// name wins and ours moves, so shared vocabulary means the same thing on both
/// sides. Do not reuse `ProgressSink` for a display type.
///
/// Per-file utterance counts reach this trait through the ordinary
/// [`Self::update`] path, as `progress_current` / `progress_total` on a file
/// entry. There is no separate job-level batch method: the aggregate that used
/// one was retired on 2026-07-30 (see `runner::util::batch_progress`).
pub trait ProgressDisplay: Send + Sync {
    /// Update completed file count and file status entries.
    fn update(&self, done: u64, file_statuses: &[FileStatusEntry]);
    /// Log a successfully completed file, with what its producer left out
    /// on purpose (information, listed whatever the file's standing).
    fn log_done(&self, filename: &str, exclusions: &[crate::api::OutputExclusionRecord]);
    /// Log a failed file with error message.
    fn log_error(&self, filename: &str, msg: &str);
    /// Log a file whose output was written with admission diagnostics:
    /// neither a clean success nor a failure. `None` when the server did not
    /// record them (a status restored from an older database row).
    fn log_diagnosed(
        &self,
        filename: &str,
        diagnostics: Option<&crate::api::FileOutputDiagnostics>,
        exclusions: &[crate::api::OutputExclusionRecord],
    );
    /// Mark processing as complete.
    fn finish(&self);
    /// Update server health snapshot. Default no-op for non-TUI sinks.
    fn update_health(&self, _health: &HealthResponse) {}
    /// Surface a cancellation receipt for the end-of-run banner.
    /// Default no-op for non-TUI sinks (which already print the
    /// receipt to stderr inline as part of their finish() output).
    fn send_cancelled_receipt(&self, _receipt: crate::cli::tui::app::CancelledReceipt) {}
}

/// Progress display for batch processing, overall bar + activity spinner.
///
/// Shows:
/// ```text
///   [=====>                  ] 3/50 files  [00:42]
///   ⠋ morphotag: stanza processing
/// ```
pub struct BatchProgress {
    mp: indicatif::MultiProgress,
    overall: ProgressBar,
    activity: ProgressBar,
    command: String,
}

impl BatchProgress {
    /// Create a new batch progress display.
    pub fn new(total: u64, command: &str) -> Self {
        let mp = indicatif::MultiProgress::new();

        let overall = mp.add(ProgressBar::new(total));
        // indicatif template strings are validated at compile time
        // by the `template(...)` parser; the literal here is fixed.
        #[allow(clippy::expect_used)]
        overall.set_style(
            ProgressStyle::default_bar()
                .template("  [{bar:30.cyan/dim}] {pos}/{len} files  [{elapsed_precise}]")
                .expect("valid template")
                .progress_chars("=>-"),
        );
        overall.set_position(0);

        let activity = mp.add(ProgressBar::new_spinner());
        // Same template-literal invariant.
        #[allow(clippy::expect_used)]
        activity.set_style(
            ProgressStyle::default_spinner()
                .template("  {spinner:.blue} {msg}")
                .expect("valid template"),
        );
        activity.enable_steady_tick(std::time::Duration::from_millis(120));

        Self {
            mp,
            overall,
            activity,
            command: command.to_string(),
        }
    }

    /// Update completed file count and activity from file status entries.
    pub fn update(&self, done: u64, file_statuses: &[FileStatusEntry]) {
        self.overall.set_position(done);

        // Find a file with an active progress label to show activity
        if let Some(active) = file_statuses
            .iter()
            .find(|f| f.progress_label.is_some() && f.status == FileStatusKind::Processing)
        {
            let label = active.progress_label.as_deref().unwrap_or("processing");
            let pct = match (active.progress_current, active.progress_total) {
                (Some(c), Some(t)) if t > 0 => format!(" ({c}/{t})"),
                _ => String::new(),
            };
            self.activity
                .set_message(format!("{}: {label}{pct}", self.command));
        }
    }

    /// Log a successfully completed file (printed above the progress bar),
    /// then each thing its producer left out on purpose.
    pub fn log_done(&self, filename: &str, exclusions: &[crate::api::OutputExclusionRecord]) {
        self.overall.println(format!("  \u{2713} {filename}"));
        self.log_exclusions(exclusions);
    }

    /// Log a file written with diagnostics (printed above the progress bar),
    /// then each thing its producer left out on purpose.
    pub fn log_diagnosed(
        &self,
        filename: &str,
        diagnostics: Option<&crate::api::FileOutputDiagnostics>,
        exclusions: &[crate::api::OutputExclusionRecord],
    ) {
        self.overall.println(format!(
            "  ! {filename}: {}",
            diagnosed_summary(diagnostics)
        ));
        self.log_exclusions(exclusions);
    }

    /// One indented line per exclusion, under its file's line.
    fn log_exclusions(&self, exclusions: &[crate::api::OutputExclusionRecord]) {
        for exclusion in exclusions {
            self.overall.println(format!("    {exclusion}"));
        }
    }

    /// Log a failed file (printed above the progress bar).
    pub fn log_error(&self, filename: &str, msg: &str) {
        let first_line = msg.split('\n').next().unwrap_or("unknown error");
        self.overall
            .println(format!("  \u{2717} {filename}: {first_line}"));
    }

    /// Mark processing as complete and clear the bars.
    pub fn finish(&self) {
        self.activity.finish_and_clear();
        self.overall.finish_and_clear();
        // Force clear the multi-progress to avoid ghost lines
        let _ = &self.mp;
    }
}

impl ProgressDisplay for BatchProgress {
    fn update(&self, done: u64, file_statuses: &[FileStatusEntry]) {
        self.update(done, file_statuses);
    }

    fn log_done(&self, filename: &str, exclusions: &[crate::api::OutputExclusionRecord]) {
        self.log_done(filename, exclusions);
    }

    fn log_error(&self, filename: &str, msg: &str) {
        self.log_error(filename, msg);
    }

    fn log_diagnosed(
        &self,
        filename: &str,
        diagnostics: Option<&crate::api::FileOutputDiagnostics>,
        exclusions: &[crate::api::OutputExclusionRecord],
    ) {
        self.log_diagnosed(filename, diagnostics, exclusions);
    }

    fn finish(&self) {
        self.finish();
    }
}

/// One line describing a file written with diagnostics: how many findings and
/// the first of them, how many stages were not applied, and how many words
/// forced alignment left untimed.
pub(crate) fn diagnosed_summary(diagnostics: Option<&crate::api::FileOutputDiagnostics>) -> String {
    let Some(diagnostics) = diagnostics else {
        return "written with diagnostics (not recorded)".to_string();
    };
    let count = diagnostics.finding_count();
    let noun = if count == 1 {
        "diagnostic"
    } else {
        "diagnostics"
    };
    let mut line = format!("written with {count} {noun}");
    if let Some(first) = diagnostics.first_findings().first() {
        line.push_str(&format!(" (first: {first})"));
    }
    let mut stages = 0;
    for shortfall in &diagnostics.shortfalls {
        match shortfall {
            crate::api::OutputShortfallRecord::StageSkipped { .. }
            | crate::api::OutputShortfallRecord::StageNotApplied { .. } => stages += 1,
            crate::api::OutputShortfallRecord::StageHeldOut {
                stage,
                held_out_utterances,
                ..
            } => line.push_str(&format!(
                "; {} left out of {held_out_utterances} utterance(s)",
                stage.name()
            )),
            crate::api::OutputShortfallRecord::TimingIncomplete {
                required_words,
                untimed_words,
                ..
            } => line.push_str(&format!(
                "; {untimed_words} of {required_words} words untimed"
            )),
        }
    }
    if stages > 0 {
        line.push_str(&format!("; {stages} requested stage(s) not applied"));
    }
    line
}
