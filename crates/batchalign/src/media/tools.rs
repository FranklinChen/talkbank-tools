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
    ///
    /// In this crate's own test builds, the first spawn of the process first
    /// checks that the tools on `PATH` are the pinned release
    /// ([`Self::require_pinned_release`]). Every media spawn goes through here
    /// or [`Self::async_command`], the production code under test included, so
    /// no unit test can generate or decode audio with an unpinned ffmpeg and
    /// no test has to remember to ask. Integration tests link the non-test
    /// build and call the check themselves.
    #[must_use]
    pub fn command(self) -> std::process::Command {
        #[cfg(test)]
        pin_check::ensure();
        self.unchecked_command()
    }

    /// An async `Command` already naming this tool; pin-checked in test
    /// builds exactly as [`Self::command`] is.
    #[must_use]
    pub fn async_command(self) -> tokio::process::Command {
        #[cfg(test)]
        pin_check::ensure();
        tokio::process::Command::new(self.program())
    }

    /// The `Command` with no test-build pin check: for [`Self::banner`],
    /// which the pin check itself calls to read the release.
    fn unchecked_command(self) -> std::process::Command {
        std::process::Command::new(self.program())
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
            .unchecked_command()
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
            .kill_on_drop(true)
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

/// The release pin, read at compile time from the one file CI and the local
/// gate also read, so the tests cannot disagree with them about the release.
const PIN_FILE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../scripts/ffmpeg-pin.sh"
));

/// The value of the `FFMPEG_VERSION=` line in the text of `ffmpeg-pin.sh`.
///
/// `None` when no such line exists; the caller turns that into a loud failure
/// rather than a default, because a pin that silently became "anything" would
/// let every media test run against the wrong decoder again.
fn pinned_version(pin_file: &str) -> Option<&str> {
    pin_file
        .lines()
        .find_map(|line| line.strip_prefix("FFMPEG_VERSION="))
        .map(str::trim)
        .filter(|version| !version.is_empty())
}

/// Why the tools on `PATH` are not the pinned release: the three ways the
/// test-support check can fail, each naming what to fix.
#[doc(hidden)]
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum PinMismatch {
    /// The pin file itself states no release.
    #[error("scripts/ffmpeg-pin.sh has no FFMPEG_VERSION= line")]
    NoPin,
    /// The tool runs, but it is another release.
    #[error(
        "{tool} reports {banner:?}, but the media tests are pinned to {pin} (scripts/ffmpeg-pin.sh)"
    )]
    WrongRelease {
        /// `ffmpeg` or `ffprobe`.
        tool: &'static str,
        /// The first line of its `-version` output.
        banner: String,
        /// The pinned release.
        pin: &'static str,
    },
    /// The tool cannot be run at all.
    #[error(
        "{tool} is not runnable, and the media tests need release {pin} (scripts/ffmpeg-pin.sh)"
    )]
    NotRunnable {
        /// `ffmpeg` or `ffprobe`.
        tool: &'static str,
        /// The pinned release.
        pin: &'static str,
    },
}

impl MediaTool {
    /// TEST SUPPORT: whether BOTH `ffmpeg` and `ffprobe` on `PATH` are the
    /// release pinned in `scripts/ffmpeg-pin.sh`.
    ///
    /// Every media test generates audio with ffmpeg's encoders and compares what
    /// ffprobe states with what ffmpeg decodes, so its answers are properties
    /// of the RELEASE (ffmpeg 6.1 decodes an AAC-in-MP4 tone 17.5 ms longer
    /// than 9.0 does). The shell gate `scripts/check-ffmpeg-pin.sh` enforces
    /// the pin for `make` and CI, but a bare `cargo test` bypasses it; this is
    /// the same check, made by the tests themselves, with the same comparison
    /// (first banner line starts with `<tool> version <pin> `).
    ///
    /// Returns the mismatch rather than panicking: library code never panics,
    /// and a test turns an `Err` into its failure. In this crate's test builds
    /// [`Self::command`] makes that call itself; integration tests make it
    /// explicitly. Checked once per process, since every spawn asks.
    ///
    /// `pub` and `doc(hidden)` only because integration tests under `tests/`
    /// cannot see `#[cfg(test)]` items; it is not a production API.
    #[doc(hidden)]
    pub fn require_pinned_release() -> Result<(), PinMismatch> {
        static CHECKED: std::sync::OnceLock<Result<(), PinMismatch>> = std::sync::OnceLock::new();
        CHECKED
            .get_or_init(|| {
                let pin = pinned_version(PIN_FILE).ok_or(PinMismatch::NoPin)?;
                for tool in [Self::Ffmpeg, Self::Ffprobe] {
                    let expected = format!("{} version {pin} ", tool.program());
                    // The trailing space in `expected` matches the gate: a banner
                    // is followed by more text, and `9.0.2` must not match `9.0.21`.
                    match tool.banner() {
                        Some(banner) if banner.starts_with(&expected) => {}
                        Some(banner) => {
                            return Err(PinMismatch::WrongRelease {
                                tool: tool.program(),
                                banner,
                                pin,
                            });
                        }
                        None => {
                            return Err(PinMismatch::NotRunnable {
                                tool: tool.program(),
                                pin,
                            });
                        }
                    }
                }
                Ok(())
            })
            .clone()
    }
}

/// The test-build half of the spawn-path check: a test that spawns a media
/// tool fails, with the mismatch's own message, unless the pinned release is
/// on `PATH`. A `cfg(test)` module, so the failure is test code's to raise.
#[cfg(test)]
mod pin_check {
    pub(super) fn ensure() {
        if let Err(mismatch) = super::MediaTool::require_pinned_release() {
            panic!("{mismatch}");
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

    /// The pin is read out of the real `scripts/ffmpeg-pin.sh`, and a text with
    /// no `FFMPEG_VERSION=` line (or an empty one) has no pin.
    #[test]
    fn the_pin_is_read_from_the_pin_file() {
        let pin = pinned_version(PIN_FILE).expect("the real pin file states a version");
        assert!(
            pin.chars().all(|c| c.is_ascii_digit() || c == '.') && pin.contains('.'),
            "not a dotted release number: {pin:?}"
        );
        assert_eq!(
            pinned_version("x\nFFMPEG_VERSION=9.0.2\ny\n"),
            Some("9.0.2")
        );
        assert_eq!(pinned_version("# no pin here\nOTHER=1\n"), None);
        assert_eq!(pinned_version("FFMPEG_VERSION=\n"), None);
    }

    /// A tool that RAN and failed is `Ok`. Only failing to reach it is an error.
    ///
    /// This is the half of `run`'s contract that a caller most easily gets
    /// wrong: every production site distinguishes "ffmpeg is missing" from
    /// "ffmpeg rejected this input", and collapsing them would report a broken
    /// media file as an uninstalled binary.
    #[test]
    fn a_tool_that_ran_and_failed_is_not_an_error() {
        let output = MediaTool::Ffmpeg
            .run(["-nonsense-flag-that-does-not-exist"])
            .expect("ffmpeg is installed, so running it must not be a spawn error");
        assert!(!output.status.success(), "the flag should be rejected");
    }
}
