//! The external media tools this crate shells out to.
//!
//! ONE statement of each tool's name, and one construction path for every
//! spawn of it. The names were previously written as string literals at every
//! call site across six modules, and `ffmpeg`'s availability predicate existed
//! four times: the real one in `ensure_wav`, a delegating shim in
//! `artifacts_v2`, and two independent copies in test code. `ffprobe` had no
//! predicate at all, so `doctor` open-coded its own spawn to ask the question.
//!
//! Nothing here proves a tool WORKS, and the distinction is load-bearing.
//! `ffmpeg` can be installed, on `PATH`, and answer `-version` correctly while
//! every actual decode fails: the decoding path pulls in separately-versioned
//! shared libraries, so a system upgrade can break decoding without touching
//! the binary this module spawns. A capability token minted from `available()`
//! would therefore read `true` on exactly the machines that cannot process
//! audio, which is a label wearing a proof's clothes. Presence is also a fact
//! about the world at the moment it is probed, and the world can change before
//! the spawn.
//!
//! So availability answers "can this be run", and nothing more. Proof that a
//! machine can actually decode is a separate, stronger check that reads real
//! audio and inspects the frames it gets back.

/// A media tool this crate invokes as a subprocess.
///
/// The variants are the closed set of external programs the crate depends on.
/// Adding one here is what makes it spawnable, so a new tool cannot arrive as
/// a bare string at one call site.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MediaTool {
    /// Transcoding, segment extraction, prepared-audio materialization.
    Ffmpeg,
    /// Duration probing. Optional in practice: a host missing it can still run
    /// every conversion, which is why `doctor` reports it as detail rather
    /// than as a verdict.
    Ffprobe,
}

impl MediaTool {
    /// The program name, stated once for the whole crate.
    #[must_use]
    pub const fn program(self) -> &'static str {
        match self {
            Self::Ffmpeg => "ffmpeg",
            Self::Ffprobe => "ffprobe",
        }
    }

    /// A blocking `Command` already naming this tool.
    ///
    /// Callers add arguments; they never name the program, which is the point.
    #[must_use]
    pub fn command(self) -> std::process::Command {
        std::process::Command::new(self.program())
    }

    /// An async `Command` already naming this tool.
    #[must_use]
    pub fn async_command(self) -> tokio::process::Command {
        tokio::process::Command::new(self.program())
    }

    /// The first line of `<tool> -version`, or `None` when it cannot be run.
    ///
    /// Returns the banner rather than a bare bool because the availability
    /// probe ALREADY captures it: `Command::output()` collects stdout whether
    /// or not the caller wants it. A caller wanting both the verdict and the
    /// version pays one spawn instead of two, which `batchalign3 doctor` was
    /// doing on every healthy run before this was returned.
    ///
    /// `stdin` is closed. `ffmpeg` reads stdin when it has one and will
    /// happily consume a terminal's input; the previous `ffmpeg` probe left it
    /// inherited while the neighbouring `ffprobe` probe closed it, which is
    /// the kind of difference that survives because nothing states it once.
    #[must_use]
    pub fn banner(self) -> Option<String> {
        let output = self
            .command()
            .arg("-version")
            .stdin(std::process::Stdio::null())
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .next()
            .map(|line| line.trim().to_owned())
    }

    /// Run this tool with `args`, telling "not installed" apart from
    /// everything else that can go wrong.
    ///
    /// Visible only inside `crate::media`, so the operation types are the
    /// ONLY things that classify a spawn. That is the closure this whole
    /// module exists for: a production path cannot open-code a spawn and
    /// invent its own reading of what went wrong, because it cannot reach
    /// this. Earlier rounds made better primitives and left them public,
    /// and the duplication simply moved.
    ///
    /// This is what the operations call, and why no site needs a pre-flight
    /// availability probe: `Command::output()` already reports
    /// `ErrorKind::NotFound` for a program that is not on `PATH`, at the
    /// moment the caller actually cares about rather than a moment earlier.
    ///
    /// A non-zero exit is NOT an error here: the tool ran, and reading its
    /// status is the caller's job.
    ///
    pub(in crate::media) fn run<I, S>(self, args: I) -> Result<std::process::Output, MediaToolError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<std::ffi::OsStr>,
    {
        self.command()
            .args(args)
            .output()
            .map_err(|source| self.classify_spawn_failure(source))
    }

    /// [`Self::run`] for callers already inside an async context.
    ///
    /// A separate method rather than a separate module: the SPLIT is about
    /// where the caller runs, not about what the tool is, and the earlier
    /// design let that language-level difference become a reason for one
    /// production path to keep its own spawn and its own error handling.
    pub(in crate::media) async fn run_async<I, S>(
        self,
        args: I,
    ) -> Result<std::process::Output, MediaToolError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<std::ffi::OsStr>,
    {
        self.async_command()
            .args(args)
            .output()
            .await
            .map_err(|source| self.classify_spawn_failure(source))
    }

    /// Which kind of "could not run" an OS error represents.
    ///
    /// Split out so the mapping is reachable by a test: making `ffmpeg`
    /// genuinely absent needs either a weakened enum or a process-wide
    /// `PATH` mutation, and neither is worth it.
    fn classify_spawn_failure(self, source: std::io::Error) -> MediaToolError {
        if source.kind() == std::io::ErrorKind::NotFound {
            MediaToolError::NotInstalled(self)
        } else {
            MediaToolError::Spawn { tool: self, source }
        }
    }
}

/// Why a media tool could not be RUN.
///
/// Module-internal: consumers outside `crate::media` see the OPERATION's
/// error ([`super::transcode::TranscodeError`]), which names what was being
/// attempted, not merely which binary was involved.
///
/// Only about reaching the program. Whether it then succeeded is the caller's
/// question, and deliberately not modelled here.
#[derive(Debug, thiserror::Error)]
pub(in crate::media) enum MediaToolError {
    /// The program is not installed, or not on `PATH`.
    #[error("{} is not installed or not on PATH", .0.program())]
    NotInstalled(MediaTool),
    /// The program exists but could not be spawned (permissions, fork limits).
    #[error("could not run {}: {source}", .tool.program())]
    Spawn {
        /// The tool that could not be spawned.
        tool: MediaTool,
        /// What the operating system said.
        source: std::io::Error,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The program names, pinned through the construction path callers use.
    ///
    /// A WIRE FORMAT in the sense that matters: these strings are what reaches
    /// `execvp`, and no type can hold what a PATH lookup will accept. Asserted
    /// against the literals rather than against `program()`, because a test
    /// that compares `command()` to `program()` compares a two-line
    /// constructor with its own argument and can only fail if someone rewrites
    /// it to name a literal.
    #[test]
    fn each_tool_spawns_the_program_it_names() {
        assert_eq!(MediaTool::Ffmpeg.command().get_program(), "ffmpeg");
        assert_eq!(MediaTool::Ffprobe.command().get_program(), "ffprobe");
        assert_eq!(
            MediaTool::Ffmpeg.async_command().as_std().get_program(),
            "ffmpeg"
        );
    }

    /// A missing program is `NotInstalled`, everything else is a spawn failure.
    ///
    /// POLICY, not an invariant: `ErrorKind::NotFound` is what the OS reports
    /// for a program that is not on `PATH`, and treating exactly that as "not
    /// installed" is a choice with alternatives. It is the choice that lets
    /// production drop its pre-flight `-version` probe, so it is worth pinning.
    #[test]
    fn only_a_missing_program_reads_as_not_installed() {
        let missing = MediaTool::Ffmpeg
            .classify_spawn_failure(std::io::Error::from(std::io::ErrorKind::NotFound));
        assert!(matches!(
            missing,
            MediaToolError::NotInstalled(MediaTool::Ffmpeg)
        ));

        let denied = MediaTool::Ffprobe
            .classify_spawn_failure(std::io::Error::from(std::io::ErrorKind::PermissionDenied));
        assert!(matches!(
            denied,
            MediaToolError::Spawn {
                tool: MediaTool::Ffprobe,
                ..
            }
        ));
    }

    /// A tool that RAN and failed is `Ok`. Only failing to reach it is an error.
    ///
    /// This is the half of `run`'s contract that a caller most easily gets
    /// wrong: every production site distinguishes "ffmpeg is missing" from
    /// "ffmpeg rejected this input", and collapsing them would report a broken
    /// media file as an uninstalled binary.
    #[test]
    fn a_tool_that_ran_and_failed_is_not_an_error() {
        // Required, not optional: the gate pins ffmpeg before tests run, and a
        // test that returned early without it would read as a pass.
        let output = MediaTool::Ffmpeg
            .run(["-nonsense-flag-that-does-not-exist"])
            .expect("ffmpeg is installed, so running it must not be a spawn error");
        assert!(!output.status.success(), "the flag should be rejected");
    }
}
