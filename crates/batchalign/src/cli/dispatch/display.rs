//! What a submission shows: whether the browser opens on the job's dashboard
//! page, decided once per invocation from the flags, the session and how the
//! server was come by.
//!
//! The rule (see [`dashboard_launch`]): the page opens only in an interactive
//! session, never under `--no-open-dashboard` or the `BATCHALIGN_NO_BROWSER`
//! backstop, and by default only on the invocation that started the daemon.
//! Before this module every submission opened a tab (the flag that could stop
//! it was the only one that acted), so a shell loop over ten directories
//! opened ten.

use crate::cli::args::Dashboard;
use crate::cli::daemon::ServerOrigin;

/// Whether the invocation is talking to a person at a terminal.
///
/// Read once from stdout, where the TUI would draw: a cron job, a CI step, a
/// piped command or a script is not a session a browser should open into, and
/// the TUI is not a display it can use either.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Session {
    /// stdout is a terminal.
    Interactive,
    /// stdout is a pipe, a file or nothing.
    NonInteractive,
}

impl Session {
    /// The session this process runs in.
    pub(super) fn of_stdout() -> Self {
        use std::io::IsTerminal;
        if std::io::stdout().is_terminal() {
            Self::Interactive
        } else {
            Self::NonInteractive
        }
    }
}

/// The hidden backstop for tests and harnesses that must never see a browser:
/// the `BATCHALIGN_NO_BROWSER` environment variable, present or not.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum BrowserEnv {
    /// The variable is not set.
    Allowed,
    /// The variable is set (to anything).
    Suppressed,
}

impl BrowserEnv {
    /// Read the backstop from this process's environment.
    pub(super) fn of_env() -> Self {
        if std::env::var_os("BATCHALIGN_NO_BROWSER").is_some() {
            Self::Suppressed
        } else {
            Self::Allowed
        }
    }
}

/// The decision: whether this submission's dashboard page opens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DashboardLaunch {
    /// Open the page in the local browser (macOS; other platforms print the
    /// URL, which every path does anyway).
    Launch,
    /// Print the URL only.
    Skip,
}

/// Decide whether the dashboard opens, from the four facts that bear on it.
///
/// `Closed`, a non-interactive session and the environment backstop each
/// forbid it. `Open` launches in any interactive session, since the operator
/// asked for this submission's page. With neither flag the page opens once,
/// on the invocation that started the daemon: an already-running server has
/// had its page opened by whoever started it, or the operator shut it.
pub(super) fn dashboard_launch(
    dashboard: Dashboard,
    origin: ServerOrigin,
    session: Session,
    browser: BrowserEnv,
) -> DashboardLaunch {
    match (dashboard, origin, session, browser) {
        (Dashboard::Closed, _, _, _)
        | (_, _, Session::NonInteractive, _)
        | (_, _, _, BrowserEnv::Suppressed) => DashboardLaunch::Skip,
        (Dashboard::Open, _, Session::Interactive, BrowserEnv::Allowed) => DashboardLaunch::Launch,
        (
            Dashboard::OnDaemonStart,
            ServerOrigin::StartedByThisInvocation,
            Session::Interactive,
            BrowserEnv::Allowed,
        ) => DashboardLaunch::Launch,
        (
            Dashboard::OnDaemonStart,
            ServerOrigin::AlreadyRunning,
            Session::Interactive,
            BrowserEnv::Allowed,
        ) => DashboardLaunch::Skip,
    }
}

/// Open the submitted job's dashboard page when the decision says so.
///
/// Only macOS launches a browser (`open`); every other target prints nothing
/// here, since the URL has already been printed beside the job id. The
/// decision itself is platform-independent and tested everywhere.
pub(super) fn open_dashboard(dashboard_url: &str, launch: DashboardLaunch) {
    match launch {
        DashboardLaunch::Skip => {}
        DashboardLaunch::Launch => {
            #[cfg(target_os = "macos")]
            {
                let _ = std::process::Command::new("open")
                    .arg(dashboard_url)
                    .stdout(std::process::Stdio::null())
                    .stderr(std::process::Stdio::null())
                    .spawn();
            }
            #[cfg(not(target_os = "macos"))]
            {
                let _ = dashboard_url;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const INTERACTIVE: (Session, BrowserEnv) = (Session::Interactive, BrowserEnv::Allowed);

    fn launch(dashboard: Dashboard, origin: ServerOrigin) -> DashboardLaunch {
        dashboard_launch(dashboard, origin, INTERACTIVE.0, INTERACTIVE.1)
    }

    /// `--open-dashboard` acts: it opens even when the daemon was already
    /// running, which the default would not.
    #[test]
    fn the_positive_flag_opens_on_a_reused_daemon() {
        assert_eq!(
            launch(Dashboard::Open, ServerOrigin::AlreadyRunning),
            DashboardLaunch::Launch
        );
        assert_eq!(
            launch(Dashboard::Open, ServerOrigin::StartedByThisInvocation),
            DashboardLaunch::Launch
        );
    }

    /// `--no-open-dashboard` closes, even on the invocation that started it.
    #[test]
    fn the_negative_flag_never_opens() {
        for origin in [
            ServerOrigin::StartedByThisInvocation,
            ServerOrigin::AlreadyRunning,
        ] {
            assert_eq!(launch(Dashboard::Closed, origin), DashboardLaunch::Skip);
        }
    }

    /// With neither flag, the page opens once: on the invocation that started
    /// the daemon, and not on the ones that reuse it.
    #[test]
    fn by_default_a_reused_daemon_does_not_open_and_a_started_one_does() {
        assert_eq!(
            launch(Dashboard::OnDaemonStart, ServerOrigin::AlreadyRunning),
            DashboardLaunch::Skip
        );
        assert_eq!(
            launch(
                Dashboard::OnDaemonStart,
                ServerOrigin::StartedByThisInvocation
            ),
            DashboardLaunch::Launch
        );
    }

    /// A non-interactive stdout (cron, CI, a pipe) never opens, whatever the
    /// flag; nor does the environment backstop.
    #[test]
    fn a_non_interactive_session_or_the_env_backstop_never_opens() {
        for dashboard in [Dashboard::Open, Dashboard::OnDaemonStart] {
            assert_eq!(
                dashboard_launch(
                    dashboard,
                    ServerOrigin::StartedByThisInvocation,
                    Session::NonInteractive,
                    BrowserEnv::Allowed,
                ),
                DashboardLaunch::Skip
            );
            assert_eq!(
                dashboard_launch(
                    dashboard,
                    ServerOrigin::StartedByThisInvocation,
                    Session::Interactive,
                    BrowserEnv::Suppressed,
                ),
                DashboardLaunch::Skip
            );
        }
    }
}
