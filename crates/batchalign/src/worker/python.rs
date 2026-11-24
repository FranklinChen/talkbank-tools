//! Python runtime resolution for worker subprocesses.
//!
//! Resolution order:
//! 1. `BATCHALIGN_PYTHON` (explicit override)
//! 2. `VIRTUAL_ENV` interpreter (preferring Python 3.14 names where available)
//! 3. Sibling python in the same directory as the binary (pip-installed case)
//! 4. Walk up from the binary looking for a `.venv` that has batchalign
//! 5. `python3.14` on Unix-like systems, or `python` on Windows

use std::collections::HashSet;
use std::ffi::OsStr;
use std::path::{Path, PathBuf};

/// The Python statement every worker process runs.
const WORKER_ENTRY: &str = "import sys; sys.argv = ['batchalign-worker'] + sys.argv[1:]; from batchalign.worker import main; main()";

/// `python -P`: an interpreter that does not put the current directory on
/// `sys.path`.
///
/// Why: `python -c` prepends the working directory to `sys.path`, so a worker
/// started by a server whose working directory is a source checkout imported
/// `batchalign` from that checkout's working tree, uncommitted edits included,
/// instead of the package installed beside the interpreter, while the build
/// identity reported the installed build (observed 2026-10-01). `-P` (Python
/// 3.11+; we require 3.14) removes that entry and nothing else, so an editable
/// development install, which reaches its source through a `.pth` file, still
/// works.
pub(crate) fn isolated_python(python: impl AsRef<OsStr>) -> std::process::Command {
    let mut cmd = std::process::Command::new(python);
    cmd.arg("-P");
    cmd
}

/// The command that starts a worker: [`isolated_python`] running the worker
/// entry point. Every route that starts a worker builds it here.
pub(crate) fn worker_command(python: impl AsRef<OsStr>) -> std::process::Command {
    let mut cmd = isolated_python(python);
    cmd.arg("-c").arg(WORKER_ENTRY);
    cmd
}

/// The arguments that bind a stdio worker to the process launching it: the
/// worker watches this PID and exits with it. The worker refuses a stdio
/// launch without them, so every stdio launcher appends exactly these.
pub(crate) fn supervised_stdio_args() -> [String; 2] {
    [
        "--supervisor-pid".to_owned(),
        std::process::id().to_string(),
    ]
}

/// Runtime-owned inputs for resolving the worker Python executable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PythonRuntime {
    explicit_python: Option<String>,
    venv_dir: Option<PathBuf>,
    current_exe: Option<PathBuf>,
}

impl PythonRuntime {
    /// Resolve Python runtime inputs from ambient environment variables and the
    /// current executable path.
    pub fn from_env() -> Self {
        let explicit_python = std::env::var("BATCHALIGN_PYTHON").ok();
        let venv_dir = std::env::var("VIRTUAL_ENV").ok();
        let current_exe = std::env::current_exe().ok();
        Self::from_sources(
            explicit_python.as_deref(),
            venv_dir.as_deref().map(Path::new),
            current_exe.as_deref(),
        )
    }

    /// Build the runtime inputs from explicit sources.
    pub fn from_sources(
        explicit_python: Option<&str>,
        venv_dir: Option<&Path>,
        current_exe: Option<&Path>,
    ) -> Self {
        Self {
            explicit_python: explicit_python
                .map(str::trim)
                .filter(|path| !path.is_empty())
                .map(ToOwned::to_owned),
            venv_dir: venv_dir.map(Path::to_path_buf),
            current_exe: current_exe.map(Path::to_path_buf),
        }
    }

    /// Resolve the Python executable from the owned runtime inputs.
    pub fn resolve_executable(&self) -> String {
        if let Some(path) = &self.explicit_python {
            return path.clone();
        }

        if let Some(candidate) = self
            .venv_dir
            .as_deref()
            .into_iter()
            .flat_map(venv_python_candidates)
            .find(|candidate| candidate.is_file())
        {
            return candidate.to_string_lossy().into_owned();
        }

        WorkerImportSearch::from_binary(self.current_exe.as_deref())
            .resolve()
            .map_or_else(
                || default_python_command().to_owned(),
                |path| path.to_string_lossy().into_owned(),
            )
    }
}

fn venv_python_candidates(venv_dir: &Path) -> impl Iterator<Item = PathBuf> + use<> {
    let bin = venv_dir.join(if cfg!(windows) { "Scripts" } else { "bin" });
    python_candidates(bin)
}

fn python_candidates(bin: PathBuf) -> impl Iterator<Item = PathBuf> {
    sibling_python_names()
        .iter()
        .map(move |name| bin.join(name))
}

/// One ordered search for an interpreter which imports the isolated worker.
/// A path can be both beside the server and inside an ancestor's .venv.
struct WorkerImportSearch<'a> {
    bin: Option<&'a Path>,
}

impl<'a> WorkerImportSearch<'a> {
    fn from_binary(executable: Option<&'a Path>) -> Self {
        Self {
            bin: executable.and_then(Path::parent),
        }
    }

    fn resolve(self) -> Option<PathBuf> {
        self.find_with(|candidate| candidate.is_file() && python_has_batchalign(candidate))
    }

    fn find_with(self, mut admits_worker: impl FnMut(&Path) -> bool) -> Option<PathBuf> {
        let adjacent = self
            .bin
            .into_iter()
            .flat_map(|bin| python_candidates(bin.to_owned()));
        let ancestor_venvs = self
            .bin
            .into_iter()
            .flat_map(Path::ancestors)
            .flat_map(|dir| venv_python_candidates(&dir.join(".venv")));
        let mut inspected = HashSet::new();
        adjacent
            .chain(ancestor_venvs)
            .find(|candidate| inspected.insert(candidate.clone()) && admits_worker(candidate))
    }
}

/// Check whether a Python executable can import `batchalign.worker`.
fn python_has_batchalign(python: &Path) -> bool {
    isolated_python(python)
        .args(["-c", "import batchalign.worker"])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

const fn sibling_python_names() -> &'static [&'static str] {
    if cfg!(windows) {
        &["python.exe"]
    } else {
        &["python3.14", "python3", "python"]
    }
}

const fn default_python_command() -> &'static str {
    if cfg!(windows) {
        "python"
    } else {
        "python3.14"
    }
}

/// Resolve Python executable for worker subprocesses.
#[must_use]
pub fn resolve_python_executable() -> String {
    PythonRuntime::from_env().resolve_executable()
}

#[cfg(test)]
mod tests {
    /// Guards: The worker command uses the packaged entry point and excludes the
    /// current directory from Python's import path.
    /// Kind: contract
    /// Not a type: The host Python interpreter decides whether its actual
    /// import search path includes the process working directory.
    /// State: One real interpreter and a temporary import marker with a positive control.
    /// Concurrency: Commands run in their own directories without altering the parent.
    #[test]
    fn isolated_python_does_not_import_from_the_working_directory() {
        let cwd = tempfile::tempdir().expect("tmp");
        let module = cwd.path().join("ba3_cwd_probe");
        std::fs::create_dir(&module).expect("mkdir");
        std::fs::write(module.join("__init__.py"), b"").expect("write");
        let imports = |mut cmd: std::process::Command| {
            cmd.args(["-c", "import ba3_cwd_probe"])
                .current_dir(cwd.path())
                .stderr(std::process::Stdio::null())
                .status()
                .expect("run python3")
                .success()
        };
        assert!(
            imports(std::process::Command::new("python3")),
            "control: plain python -c imports from the working directory"
        );
        assert!(!imports(isolated_python("python3")));
        let cmd = worker_command("python3");
        let args: Vec<_> = cmd.get_args().collect();
        assert_eq!(
            args,
            [OsStr::new("-P"), OsStr::new("-c"), OsStr::new(WORKER_ENTRY)]
        );
    }

    use super::*;

    /// Guards: Explicit Python overrides a present active venv; absent inputs
    /// use the platform command.
    /// Kind: policy
    /// Not a type: A valid interpreter path from each source still leaves
    /// precedence and fallback spelling to the runtime resolution policy.
    /// State: Only active-venv file existence is observed in an owned directory.
    /// Concurrency: No environment variables or interpreter installations change.
    #[test]
    fn python_resolution_prefers_explicit_then_active_venv_then_baseline() {
        let tmp = tempfile::tempdir().expect("tmp");
        let sub = if cfg!(windows) {
            tmp.path().join("Scripts")
        } else {
            tmp.path().join("bin")
        };
        std::fs::create_dir_all(&sub).expect("mkdir");
        let py = if cfg!(windows) {
            sub.join("python.exe")
        } else {
            sub.join("python3.14")
        };
        std::fs::write(&py, b"").expect("touch");

        let from_sources = |explicit| {
            PythonRuntime::from_sources(explicit, Some(tmp.path()), None).resolve_executable()
        };
        assert_eq!(from_sources(Some("  /x/python  ")), "/x/python");
        assert_eq!(from_sources(Some("  ")), py.to_string_lossy());
        let fallback = PythonRuntime::from_sources(None, None, None).resolve_executable();
        assert_eq!(
            fallback,
            if cfg!(windows) {
                "python"
            } else {
                "python3.14"
            }
        );
    }

    /// Guards: Overlapping discovery paths attempt each interpreter once.
    /// Kind: policy
    /// Not a type: An executable can occur in two directory roles, and the
    /// search order controls which admission attempt runs and whether it repeats.
    /// State: Logical paths and a probe recorder, with no interpreter launched.
    /// Concurrency: The recorder and search are owned by this case.
    #[test]
    fn overlapping_python_discovery_probes_each_interpreter_once() {
        let project = Path::new("project");
        let venv = project.join(".venv");
        let preferred = venv_python_candidates(&venv)
            .next()
            .expect("platform Python");
        let executable = preferred.with_file_name("batchalign3");
        let mut observations = Vec::new();
        assert_eq!(
            WorkerImportSearch::from_binary(Some(&executable)).find_with(|candidate| {
                observations.push(candidate.to_owned());
                false
            }),
            None
        );
        assert_eq!(observations.first(), Some(&preferred));
        assert_eq!(
            observations
                .iter()
                .filter(|path| *path == &preferred)
                .count(),
            1
        );
    }
}
