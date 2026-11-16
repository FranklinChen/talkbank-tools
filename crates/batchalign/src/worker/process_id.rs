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
pub(crate) struct NativePid(std::num::NonZero<RawPid>);
impl NativePid {
    pub(crate) fn admit(pid: u32) -> Option<Self> {
        #[cfg(unix)]
        let raw = RawPid::try_from(pid).ok()?;
        #[cfg(not(unix))]
        let raw = pid;
        std::num::NonZero::new(raw).map(Self)
    }
    pub(crate) fn get(self) -> u32 {
        self.0.get() as u32
    }
    #[cfg(unix)]
    pub(crate) fn is_alive(self) -> bool {
        // SAFETY: admission established a positive process ID representable by pid_t.
        unsafe { libc::kill(self.0.get(), 0) == 0 }
    }
    #[cfg(not(unix))]
    pub(crate) fn is_alive(self) -> bool {
        true
    }
    #[cfg(unix)]
    fn send(self, scope: SignalScope, signal: libc::c_int) -> std::io::Result<()> {
        // SAFETY: the opaque target is positive and representable by pid_t.
        let result = unsafe {
            match scope {
                SignalScope::Group => libc::killpg(self.0.get(), signal),
                SignalScope::Process => libc::kill(self.0.get(), signal),
            }
        };
        if result == 0 {
            Ok(())
        } else {
            Err(std::io::Error::last_os_error())
        }
    }
    #[cfg(unix)]
    pub(crate) fn terminate_process(self) -> std::io::Result<()> {
        self.send(SignalScope::Process, libc::SIGTERM)
    }
    #[cfg(unix)]
    pub(crate) fn terminate_group(self) -> std::io::Result<()> {
        self.send(SignalScope::Group, libc::SIGTERM)
    }
    #[cfg(unix)]
    pub(crate) fn kill_group(self) -> std::io::Result<()> {
        self.send(SignalScope::Group, libc::SIGKILL)
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
            let _ = self.send(SignalScope::Process, libc::SIGKILL);
        }
    }
    #[cfg(unix)]
    pub(crate) fn of_child(child: &tokio::process::Child) -> Option<Self> {
        child.id().and_then(Self::admit)
    }
}
