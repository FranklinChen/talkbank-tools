//! Launching a process in its own session, detached from ours.
//!
//! The background server, the daemon and the TCP worker daemons must survive
//! the CLI that starts them and must not receive the terminal's signals, so
//! each child calls `setsid` between `fork` and `exec`. Doing that needs
//! `CommandExt::pre_exec`, which is `unsafe` because the hook runs in a
//! forked child where only async-signal-safe calls are sound. This module is
//! the one place that holds that `unsafe`; the three launch sites used to
//! carry their own copy, each of which ignored a failing `setsid`.

/// Make `command`'s child the leader of a new session when it starts.
///
/// A failing `setsid` now fails the spawn with the system's error rather than
/// leaving the child silently attached to our session.
#[cfg(unix)]
pub fn start_in_new_session(command: &mut std::process::Command) {
    use std::os::unix::process::CommandExt;
    // Reason: `pre_exec` is the only way to run code in the child between
    // fork and exec. The hook calls only `setsid`, which is
    // async-signal-safe, and allocates nothing.
    #[allow(unsafe_code)]
    // SAFETY: see the reason above; the closure touches no shared state.
    unsafe {
        command.pre_exec(|| {
            nix::unistd::setsid()
                .map(drop)
                .map_err(std::io::Error::from)
        });
    }
}
