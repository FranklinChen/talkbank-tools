//! What a test needs from its machine, established once per test binary and
//! never skipped.
//!
//! A Rust test has no "skipped" outcome: a test that returns early reports
//! `ok`. The worker, server and live-model tests used to do exactly that when
//! Python was missing (`eprintln!("SKIP: ...")` then `return`), so a machine
//! without the worker environment reported those suites green without
//! executing a single worker, and nothing downstream could tell that run from
//! a real pass. An unmet prerequisite now FAILS the test, and the failure
//! names what is missing and the command that provides it.
//!
//! The environment is established in order, each step only reachable from
//! the one before it:
//!
//! ```text
//! repo .venv interpreter ──> imports batchalign.worker ──> extension current
//!        (NoEnvironment)        (WorkerNotImportable)       (StaleExtension)
//! ```
//!
//! The last step runs the same content-evidence check `make` runs
//! (`scripts/check_extension_freshness.py`) against the interpreter the tests
//! will spawn, so a worker test can no longer run against a compiled
//! extension built from other sources, which is how a stale extension once
//! made failures look like regressions.

#![allow(dead_code)]

use std::fmt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

/// The checkout's root, fixed at compile time rather than found by walking up
/// from whatever directory the test runner chose.
pub fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("the crate lives two levels below the checkout root")
}

/// A prerequisite this machine does not meet, with what provides it.
#[derive(Debug, Clone)]
pub enum Missing {
    /// The checkout has no development environment.
    NoEnvironment { venv: PathBuf },
    /// The environment's interpreter cannot import the worker package.
    WorkerNotImportable {
        interpreter: PathBuf,
        stderr: String,
    },
    /// The compiled extension the interpreter imports was not built from this
    /// checkout's current sources.
    StaleExtension { report: String },
    /// A module one test needs beyond the worker package.
    ModuleNotImportable {
        module: String,
        interpreter: PathBuf,
        stderr: String,
    },
    /// Less free memory than one worker needs to start.
    Memory { available_mb: u64, required_mb: u64 },
}

impl fmt::Display for Missing {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        const DEVELOP: &str = "run `make batchalign-develop` at the checkout root";
        write!(
            f,
            "PREREQUISITE MISSING (this test fails rather than skip): "
        )?;
        match self {
            Self::NoEnvironment { venv } => write!(
                f,
                "no Python development environment at {}; {DEVELOP}",
                venv.display()
            ),
            Self::WorkerNotImportable {
                interpreter,
                stderr,
            } => write!(
                f,
                "{} cannot import batchalign.worker; {DEVELOP}\n{stderr}",
                interpreter.display()
            ),
            Self::StaleExtension { report } => write!(
                f,
                "the compiled extension is not this checkout's build; {DEVELOP}\n{report}"
            ),
            Self::ModuleNotImportable {
                module,
                interpreter,
                stderr,
            } => write!(
                f,
                "{} cannot import {module}; {DEVELOP}\n{stderr}",
                interpreter.display()
            ),
            Self::Memory {
                available_mb,
                required_mb,
            } => write!(
                f,
                "{available_mb} MB free, {required_mb} MB required to start a worker; \
                 free memory on this machine or run the suite where there is enough"
            ),
        }
    }
}

/// The Python environment worker tests spawn: the checkout's own interpreter,
/// importing `batchalign.worker`, with a current extension. Only
/// [`PythonWorker::require`] makes one.
#[derive(Debug)]
pub struct PythonWorker {
    interpreter: PathBuf,
}

static PYTHON_WORKER: OnceLock<Result<PythonWorker, Missing>> = OnceLock::new();

impl PythonWorker {
    /// The environment, established on first use and shared by the binary's
    /// tests; panics with what is missing when it cannot be.
    pub fn require() -> &'static Self {
        match PYTHON_WORKER.get_or_init(Self::establish) {
            Ok(worker) => worker,
            Err(missing) => panic!("{missing}"),
        }
    }

    /// The environment, or what is missing, for a fixture that reports the
    /// failure through its own channel before panicking.
    pub fn establish_shared() -> Result<&'static Self, Missing> {
        PYTHON_WORKER
            .get_or_init(Self::establish)
            .as_ref()
            .map_err(Clone::clone)
    }

    fn establish() -> Result<Self, Missing> {
        let root = repo_root();
        let interpreter = venv_interpreter(&root)?;
        import(&interpreter, "batchalign.worker").map_err(|stderr| {
            Missing::WorkerNotImportable {
                interpreter: interpreter.clone(),
                stderr,
            }
        })?;
        extension_current(&root, &interpreter)?;
        Ok(Self { interpreter })
    }

    /// The interpreter path, as the worker configs take it.
    pub fn interpreter(&self) -> String {
        self.interpreter.to_string_lossy().into_owned()
    }

    /// Require one more module in the same environment; panics when it is
    /// not importable.
    pub fn require_module(&self, module: &str) -> &Self {
        if let Err(stderr) = import(&self.interpreter, module) {
            panic!(
                "{}",
                Missing::ModuleNotImportable {
                    module: module.to_owned(),
                    interpreter: self.interpreter.clone(),
                    stderr,
                }
            );
        }
        self
    }
}

/// The checkout's `.venv` interpreter. Only this one: the freshness evidence
/// is recorded for the checkout's environment, so an interpreter found
/// elsewhere on `PATH` could not be shown to import a current extension.
fn venv_interpreter(root: &Path) -> Result<PathBuf, Missing> {
    let venv = root.join(".venv");
    let candidates: &[&str] = if cfg!(windows) {
        &["Scripts/python.exe"]
    } else {
        &["bin/python3.14", "bin/python3", "bin/python"]
    };
    candidates
        .iter()
        .map(|candidate| venv.join(candidate))
        .find(|path| path.is_file())
        .ok_or(Missing::NoEnvironment { venv })
}

/// Import `module` with `interpreter`; the error is Python's own stderr.
fn import(interpreter: &Path, module: &str) -> Result<(), String> {
    let output = Command::new(interpreter)
        .args(["-c", &format!("import {module}")])
        .output()
        .map_err(|error| format!("could not run {}: {error}", interpreter.display()))?;
    if output.status.success() {
        Ok(())
    } else {
        Err(String::from_utf8_lossy(&output.stderr).into_owned())
    }
}

/// Run the checkout's extension-freshness check with the interpreter the
/// tests will spawn.
fn extension_current(root: &Path, interpreter: &Path) -> Result<(), Missing> {
    let output = Command::new(interpreter)
        .args(["-m", "scripts.check_extension_freshness"])
        .current_dir(root)
        .output()
        .map_err(|error| Missing::StaleExtension {
            report: format!("could not run the freshness check: {error}"),
        })?;
    if output.status.success() {
        Ok(())
    } else {
        Err(Missing::StaleExtension {
            report: format!(
                "{}{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            ),
        })
    }
}

/// Require the free memory one worker needs to start; panics when this
/// machine has less. The spawn guard would refuse the worker anyway, with an
/// error further from the cause.
pub fn require_free_memory(required_mb: u64) {
    let available_mb = batchalign::host_memory::read_system_memory()
        .expect("the test host reports its memory")
        .available_mb
        .0;
    if available_mb < required_mb {
        panic!(
            "{}",
            Missing::Memory {
                available_mb,
                required_mb,
            }
        );
    }
}

/// The interpreter for a test that spawns its own workers: the memory one
/// worker needs, and the Python worker environment. (Its pool's ledger is
/// the declared test host's own, so nothing here isolates one.)
pub fn worker_test_python() -> String {
    require_free_memory(4096);
    PythonWorker::require().interpreter()
}
