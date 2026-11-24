//! Only positive, representable native IDs may reach process or group syscalls.
#[cfg(unix)]
type RawPid = libc::pid_t;
#[cfg(not(unix))]
type RawPid = u32;

#[cfg(unix)]
enum SignalScope {
    Process,
    Group,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NativePid(std::num::NonZero<RawPid>);
impl NativePid {
    pub(crate) fn admit(pid: u32) -> Option<Self> {
        #[cfg(unix)]
        let raw = RawPid::try_from(pid).ok()?;
        #[cfg(not(unix))]
        let raw = pid;
        std::num::NonZero::new(raw).map(Self)
    }
    pub(crate) const fn get(self) -> u32 {
        self.0.get() as u32
    }
    #[cfg(unix)]
    pub(crate) fn is_alive(self) -> bool {
        // Signal 0 probes for existence without delivering anything.
        nix::sys::signal::kill(self.nix_pid(), None).is_ok()
    }
    /// The admitted ID as `nix` takes it; admission already proved it positive
    /// and representable as `pid_t`.
    #[cfg(unix)]
    const fn nix_pid(self) -> nix::unistd::Pid {
        nix::unistd::Pid::from_raw(self.0.get())
    }
    #[cfg(not(unix))]
    pub(crate) fn is_alive(self) -> bool {
        true
    }
    #[cfg(unix)]
    fn send(self, scope: SignalScope, signal: nix::sys::signal::Signal) -> std::io::Result<()> {
        let result = match scope {
            SignalScope::Group => nix::sys::signal::killpg(self.nix_pid(), signal),
            SignalScope::Process => nix::sys::signal::kill(self.nix_pid(), signal),
        };
        result.map_err(std::io::Error::from)
    }
    #[cfg(unix)]
    pub(crate) fn terminate_process(self) -> std::io::Result<()> {
        self.send(SignalScope::Process, nix::sys::signal::Signal::SIGTERM)
    }
    #[cfg(unix)]
    pub(crate) fn terminate_group(self) -> std::io::Result<()> {
        self.send(SignalScope::Group, nix::sys::signal::Signal::SIGTERM)
    }
    #[cfg(unix)]
    pub(crate) fn kill_group(self) -> std::io::Result<()> {
        self.send(SignalScope::Group, nix::sys::signal::Signal::SIGKILL)
    }
    pub(crate) fn terminate(self) -> bool {
        #[cfg(unix)]
        {
            let group = self.terminate_group();
            let process = self.terminate_process();
            group.is_ok() || process.is_ok()
        }
        #[cfg(not(unix))]
        {
            false
        }
    }
    pub(crate) fn kill(self) {
        #[cfg(unix)]
        {
            let _ = self.kill_group();
            let _ = self.send(SignalScope::Process, nix::sys::signal::Signal::SIGKILL);
        }
    }
    #[cfg(unix)]
    pub(crate) fn of_child(child: &tokio::process::Child) -> Option<Self> {
        child.id().and_then(Self::admit)
    }
}
