//! The gate that keeps `batchalign-core` pure.
//!
//! `batchalign-core` is defined by what it CANNOT reach: no async runtime, no
//! filesystem, no subprocess, no socket, no ambient environment, no SQL, no
//! Python bridge. That definition is what makes a task runner testable against a
//! mock dispatcher instead of a GPU, a model download and a jobs database, and
//! it is the whole reason the crate is being extracted.
//!
//! Nothing in rustc expresses it: `#![forbid(...)]` takes lint names, not
//! capabilities, so one `axum = { workspace = true }` added in good faith
//! eighteen months from now ends the property with every test still green. This
//! gate is what states it, and it runs in `make lint` and CI.
//!
//! # Two halves, because one instrument cannot see both kinds of violation
//!
//! 1. **The dependency half** walks `cargo tree` for crates that hand core a
//!    capability wholesale: an HTTP server, an HTTP client, SQL, pyo3, a foreign
//!    executor. Transitive acquisition is the case it exists for, and a tree
//!    walk is the only thing that sees it.
//! 2. **The source half** greps core's own `src/` for the runtime, std-I/O and
//!    ambient-environment surface: `tokio::spawn`, `std::fs`, `std::process`,
//!    `std::net`, `std::env`. These need no dependency of their own, so nothing
//!    in the tree betrays them.
//!
//! # The third layer, which is NOT this gate
//!
//! Clippy's `disallowed_methods` / `disallowed_types` ARE capability lints, and
//! `crates/batchalign-core/clippy.toml` uses them. A per-package `clippy.toml`
//! genuinely resolves inside a workspace (cargo sets `CARGO_MANIFEST_DIR` per
//! member; verified here by probe, not assumed), and because clippy resolves
//! real paths it catches what a substring scan cannot: `use std::fs::read_to_string
//! as slurp;` then `slurp(p)`, or the same function reached through a re-export.
//!
//! It does not replace this gate, for two reasons. `disallowed-*` takes exact
//! paths with no globbing, so banning `std::fs` wholesale there means
//! enumerating the module function by function, which the source half does in
//! one pattern. And no workflow in this repo runs clippy at all: it lives in
//! `make ci-full` and `make lint-affected`. So clippy is depth for a developer
//! and this gate is the thing CI actually fails on.
//!
//! # WHY the tokio question is answered in the source half and not the tree half
//!
//! This is the trap that made the first version of this gate wrong, so it is
//! written down rather than left to be rediscovered.
//!
//! Core is meant to allow `tokio::sync` (its channels and locks are
//! executor-agnostic and pull in no scheduler) while forbidding the scheduler
//! itself. The obvious implementation is to let `tokio` appear in the tree and
//! check its enabled FEATURES. **It does not work here.** Measured on this
//! workspace: with `crates/batchalign-core` declaring
//! `tokio = { features = ["sync"] }` and nothing else, `cargo tree -p
//! batchalign-core -e normal` reported tokio with
//! `full,rt,rt-multi-thread,macros,time,fs,net,process,...` and pulled `mio` in
//! underneath it.
//!
//! The cause is Cargo's feature unification: `-p` selects the ROOT of the
//! printed tree, not an independent resolution, so every feature any other
//! workspace member turns on is present. `batchalign` enables tokio's `full`,
//! therefore so does core.
//!
//! And this is not merely a reporting artifact. Unification is what actually
//! gets compiled, so core would link a tokio that HAS `rt` in it, and
//! `tokio::spawn` written inside core would compile. A manifest-level feature
//! restriction cannot enforce purity inside one workspace at all. Hence:
//! `tokio` is not in the forbidden package set, `mio` is not either (it is
//! reachable only through unification), and the scheduler is caught by the
//! source markers, which unification cannot touch.
//!
//! # Scope, stated so the next reader does not over-trust it
//!
//! - `-e normal` excludes dev- and build-dependencies deliberately. A test
//!   wanting `#[tokio::test]` does not compromise the shipped library.
//! - The source half scans `src/` only. A test that genuinely needs an executor
//!   is not testing pure logic, so it belongs in `tests/` (an integration test
//!   linking the built lib, free to use any dev-dependency) or in the engine
//!   crate. `#[tokio::test]` is allowed inline as the one common exception.
//! - Only WHOLE-LINE comments are stripped before matching. A trailing `//` is
//!   left in place because stripping it correctly means knowing whether the
//!   slashes are inside a string literal, and guessing there would silently
//!   blind the scan. Consequence: a marker mentioned in a trailing comment is a
//!   violation, and the fix is to move the prose onto its own line. Whole-line
//!   stripping is not optional: this crate's own `lib.rs` names `std::fs` and
//!   `tokio::spawn` in its module doc while containing neither.

use std::collections::BTreeSet;
use std::fmt;
use std::path::Path;
use std::process::Command;

use crate::Result;
use crate::rust_scan::walkdir;

/// The crate this gate exists to protect. Not a parameter: a gate that can be
/// pointed at an arbitrary package is a reporting tool, and the one job here is
/// to fail the build for one specific package.
const CORE_PACKAGE: &str = "batchalign-core";

/// Where that package's sources live, relative to the repo root.
const CORE_SOURCE_DIR: &str = "crates/batchalign-core/src";

/// A cargo package name as printed by `cargo tree` (`axum`, `batchalign-core`).
///
/// Newtyped because this module compares package names, feature names and
/// source markers in the same breath, and all three read identically as
/// `String` while belonging to different namespaces.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct PackageName(String);

/// A path relative to the repo root, as reported to the operator.
#[derive(Debug, Clone, PartialEq, Eq)]
struct RepoRelativePath(String);

/// A 1-based source line number, as an editor would show it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LineNumber(usize);

impl fmt::Display for LineNumber {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self(number) = self;
        write!(f, "{number}")
    }
}

/// One capability core must not acquire, and the crates that hand it over.
///
/// Keyed by CAPABILITY rather than by crate name so the failure message says
/// what was lost, not only what was added: "core logic could read job state
/// instead of being given it" tells the reader why the gate cares, where
/// "added sqlx-core" does not.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ForbiddenCapability {
    ForeignExecutor,
    HttpServer,
    HttpClient,
    Database,
    PythonBridge,
}

impl ForbiddenCapability {
    /// Every capability. `every_forbidden_capability_is_in_all` is what keeps
    /// this honest: a hand-written array cannot notice an omission by itself,
    /// and a variant missing from it silently disables its own check.
    const ALL: &'static [Self] = &[
        Self::ForeignExecutor,
        Self::HttpServer,
        Self::HttpClient,
        Self::Database,
        Self::PythonBridge,
    ];

    /// Packages whose presence in core's normal dependency tree means core has
    /// this capability.
    ///
    /// `tokio` and `mio` are deliberately absent: see the module doc on feature
    /// unification. The tokio scheduler is caught by [`SourceMarker`] instead.
    const fn packages(self) -> &'static [&'static str] {
        match self {
            Self::ForeignExecutor => &["async-std", "smol", "async-global-executor"],
            Self::HttpServer => &["axum", "axum-core", "hyper", "tower-http"],
            Self::HttpClient => &["reqwest", "ureq"],
            Self::Database => &["sqlx", "sqlx-core", "rusqlite", "libsqlite3-sys"],
            Self::PythonBridge => &["pyo3", "pyo3-ffi", "numpy"],
        }
    }

    /// What the reader loses if this capability lands in core.
    const fn consequence(self) -> &'static str {
        match self {
            Self::ForeignExecutor => "core would carry a second executor alongside tokio",
            Self::HttpServer => "the typed model would depend on the server that serves it",
            Self::HttpClient => "a pure transform could reach the network",
            Self::Database => "core logic could read job state instead of being given it",
            Self::PythonBridge => "core would need a Python interpreter to compile",
        }
    }
}

/// One ambient capability visible in source rather than in the dependency tree.
///
/// The three tokio surfaces are deliberately NOT one marker. `tokio::sync`'s
/// channels and locks are executor-agnostic, and `async fn` plus `async_trait`
/// are language and macro surface that step 6's `Dispatcher` trait requires. It
/// is the SCHEDULER and tokio's own I/O that cannot live here, so those are what
/// [`Self::AsyncRuntime`] lists and nothing else.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SourceMarker {
    AsyncRuntime,
    Filesystem,
    Subprocess,
    Network,
    Environment,
}

impl SourceMarker {
    /// Every marker; see [`ForbiddenCapability::ALL`] on why a test guards it.
    const ALL: &'static [Self] = &[
        Self::AsyncRuntime,
        Self::Filesystem,
        Self::Subprocess,
        Self::Network,
        Self::Environment,
    ];

    /// Substrings whose presence in a code line means the file uses this
    /// capability. Chosen to be unambiguous rather than exhaustive: a false
    /// positive on a pure file wastes an afternoon and teaches the next person
    /// to distrust the gate, which costs more than a gap does.
    const fn patterns(self) -> &'static [&'static str] {
        match self {
            Self::AsyncRuntime => &[
                "tokio::spawn",
                "tokio::task",
                "tokio::runtime",
                "tokio::time",
                "tokio::select",
                "spawn_blocking",
                "JoinHandle",
                "Runtime::new",
                "#[tokio::main",
            ],
            Self::Filesystem => &[
                "std::fs",
                "tokio::fs",
                "fs::read",
                "fs::write",
                "fs::create",
                "fs::remove",
                "File::open",
                "File::create",
                "read_dir",
                "read_to_string",
            ],
            Self::Subprocess => &["std::process", "tokio::process", "Command::new"],
            Self::Network => &[
                "std::net",
                "tokio::net",
                "TcpStream",
                "TcpListener",
                "UdpSocket",
            ],
            // `std::env::` and `env::` catch the runtime readers. The `env!`
            // macro is deliberately not listed: the compiler resolves it and it
            // reads nothing at run time, so it is a different capability.
            Self::Environment => &["std::env::", "env::var", "env::args", "env::set_var"],
        }
    }

    const fn consequence(self) -> &'static str {
        match self {
            Self::AsyncRuntime => "a task runner could no longer be tested without a runtime",
            Self::Filesystem => "pure logic would need a real file laid out on disk to test",
            Self::Subprocess => "a unit test would have to have the tool installed to pass",
            Self::Network => "a unit test would depend on something answering on a port",
            Self::Environment => {
                "core could not be handed different configuration, only run under it"
            }
        }
    }
}

/// The one inline exception to [`SourceMarker::AsyncRuntime`].
///
/// `#[tokio::test]` names a dev-dependency, is unambiguous as a token, and marks
/// a test rather than library behaviour. Anything beyond it (constructing a
/// `Runtime`, spawning) means the code under test is not pure, which is the
/// finding the gate exists to surface.
const ALLOWED_TEST_ATTRIBUTE: &str = "#[tokio::test";

/// One thing wrong with core.
#[derive(Debug, PartialEq, Eq)]
enum Violation {
    /// A forbidden package appears in core's normal dependency tree.
    ForbiddenPackage {
        package: PackageName,
        capability: ForbiddenCapability,
    },
    /// A source line in core uses an ambient capability.
    ImpureSource {
        path: RepoRelativePath,
        line: LineNumber,
        marker: SourceMarker,
        pattern: &'static str,
    },
}

impl Violation {
    /// The operator-facing line.
    ///
    /// For the dependency half this names the `cargo tree -i` invocation,
    /// because the gate reports WHAT was acquired and the only useful next
    /// question is which edge brought it in, which this shape of walk cannot
    /// answer without reimplementing cargo's inverse lookup.
    fn describe(&self) -> String {
        match self {
            Self::ForbiddenPackage {
                package,
                capability,
            } => {
                let PackageName(name) = package;
                format!(
                    "{CORE_PACKAGE} depends on `{name}`, so {}.\n    \
                     Find the edge with: cargo tree -p {CORE_PACKAGE} -e normal -i {name}",
                    capability.consequence()
                )
            }
            Self::ImpureSource {
                path,
                line,
                marker,
                pattern,
            } => {
                let RepoRelativePath(path) = path;
                format!(
                    "{path}:{line} uses `{pattern}`, so {}.\n    \
                     Move the capability into batchalign-engine and pass its result in.",
                    marker.consequence()
                )
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The dependency half
// ---------------------------------------------------------------------------

/// Parse one line of `cargo tree` output into the package it names.
///
/// Returns `None` for anything that is not a package line: the section headers
/// cargo prints per dependency kind, and blank lines. Tolerant of the tree
/// glyphs, the ` (path)` suffix workspace members carry, the ` (proc-macro)`
/// marker and the trailing `(*)` cargo puts on a repeat, none of which affect
/// the name.
///
/// Features are deliberately not read. `{f}` was in the format string once, on
/// the way to a feature-level tokio verdict; the module doc explains why that
/// verdict is unsound in a workspace, and parsing a column no rule consults is
/// how a reader concludes it must matter.
fn parse_tree_line(line: &str) -> Option<PackageName> {
    // `cargo tree` renders `<name> v<version>[ (<path>)][ (proc-macro)]`, so once
    // the tree drawing is stripped the first whitespace token is the name.
    let name = line
        .trim_start_matches(|c: char| {
            c.is_whitespace() || "\u{2502}\u{251c}\u{2514}\u{2500}".contains(c)
        })
        .split_whitespace()
        .next()?;
    if name.is_empty() || name.starts_with('[') {
        return None;
    }
    Some(PackageName(name.to_owned()))
}

/// Classify a whole `cargo tree` stdout capture.
///
/// The root is skipped by NAME rather than by position, since a workspace member
/// can legitimately appear more than once in its own tree, and a package cannot
/// be its own forbidden dependency.
fn classify_tree(stdout: &str) -> Vec<Violation> {
    let mut violations: Vec<Violation> = Vec::new();
    let mut reported: BTreeSet<PackageName> = BTreeSet::new();

    for package in stdout.lines().filter_map(parse_tree_line) {
        let PackageName(name) = &package;
        if name == CORE_PACKAGE {
            continue;
        }
        let Some(capability) = ForbiddenCapability::ALL
            .iter()
            .find(|capability| capability.packages().contains(&name.as_str()))
        else {
            continue;
        };
        // One violation per package, however many times the tree reaches it.
        // Kept as an explicit statement rather than folded into the condition
        // above: a set insertion hidden in the right operand of `&&` works only
        // by short-circuit order and breaks silently when someone reorders it.
        if reported.insert(package.clone()) {
            violations.push(Violation::ForbiddenPackage {
                package,
                capability: *capability,
            });
        }
    }

    violations
}

/// Resolve core's normal dependency tree.
fn resolve_tree(root: &Path) -> std::result::Result<String, String> {
    let output = Command::new(env!("CARGO"))
        .current_dir(root)
        // Neither `--no-dedupe` nor a `-f` format: only the distinct package
        // set is read, and the default deduped tree already prints every
        // package it reaches at least once.
        .args(["tree", "-p", CORE_PACKAGE, "-e", "normal"])
        .output()
        .map_err(|error| format!("could not run `cargo tree`: {error}"))?;

    if !output.status.success() {
        // The commonest cause by far is that the crate does not exist or is not
        // a workspace member, and that must fail loudly rather than read as
        // "no violations found".
        return Err(format!(
            "`cargo tree -p {CORE_PACKAGE}` failed ({}). The gate cannot pass without the crate.\n{}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }

    String::from_utf8(output.stdout)
        .map_err(|error| format!("`cargo tree` printed invalid UTF-8: {error}"))
}

// ---------------------------------------------------------------------------
// The source half
// ---------------------------------------------------------------------------

/// Whether the whole line is a comment: `//`, `///` and `//!` alike.
///
/// Deliberately NOT a "strip the comment off this line" helper. Removing a
/// TRAILING comment correctly means knowing whether the slashes sit inside a
/// string literal, and guessing there would silently blind the scan, so the
/// module doc states the consequence instead: a marker named in a trailing
/// comment is a violation, and the fix is to move the prose onto its own line.
fn is_comment_line(line: &str) -> bool {
    line.trim_start().starts_with("//")
}

/// Scan one file's text for ambient-capability markers.
///
/// Reports at most one violation per line: the first marker found is the one
/// that has to be dealt with, and a line matching three patterns is one problem,
/// not three.
fn scan_source_text(path: &RepoRelativePath, text: &str) -> Vec<Violation> {
    let mut violations: Vec<Violation> = Vec::new();

    for (index, line) in text.lines().enumerate() {
        if is_comment_line(line) || line.contains(ALLOWED_TEST_ATTRIBUTE) {
            continue;
        }
        let found = SourceMarker::ALL.iter().find_map(|marker| {
            marker
                .patterns()
                .iter()
                .find(|pattern| line.contains(**pattern))
                .map(|pattern| (*marker, *pattern))
        });
        if let Some((marker, pattern)) = found {
            violations.push(Violation::ImpureSource {
                path: path.clone(),
                line: LineNumber(index + 1),
                marker,
                pattern,
            });
        }
    }

    violations
}

/// Scan every `.rs` file under core's `src/`.
///
/// `src/` only, by design: see the module doc on where a test that needs an
/// executor belongs.
fn scan_source_tree(root: &Path) -> std::result::Result<Vec<Violation>, String> {
    let source_dir = root.join(CORE_SOURCE_DIR);
    if !source_dir.is_dir() {
        return Err(format!(
            "{CORE_SOURCE_DIR} is not a directory. The gate cannot pass without the crate."
        ));
    }

    // `rust_scan::walkdir` is xtask's shared `.rs` enumerator, the same one the
    // sibling audits use, so the skip-set stays in one place. It is sorted here
    // because it yields directory order, and a gate whose findings move around
    // between runs is hard to diff.
    let mut paths = walkdir(&source_dir);
    paths.sort();

    let mut violations: Vec<Violation> = Vec::new();
    for path in paths {
        // The walk was rooted under `root`, so stripping cannot fail. If it ever
        // did, reporting the absolute path is strictly better than dropping the
        // finding, which is why this is a fallback rather than a `?`.
        let relative = path.strip_prefix(root).unwrap_or(&path);
        let text = std::fs::read_to_string(&path)
            .map_err(|error| format!("could not read {}: {error}", relative.display()))?;
        violations.extend(scan_source_text(
            &RepoRelativePath(relative.display().to_string()),
            &text,
        ));
    }

    Ok(violations)
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Run both halves and report everything at once.
///
/// Deliberately not short-circuiting after the dependency half: a contributor
/// who has to run the gate three times to see three problems starts running it
/// less often.
pub fn run(root: &Path) -> Result<()> {
    let mut violations = classify_tree(&resolve_tree(root)?);
    violations.extend(scan_source_tree(root)?);

    if violations.is_empty() {
        println!(
            "core-purity: {CORE_PACKAGE} has no runtime, filesystem, subprocess, socket, \
             environment, SQL or Python surface"
        );
        return Ok(());
    }

    let mut message = format!(
        "core-purity: {} violation(s). {CORE_PACKAGE} exists to be reachable without a runtime, a \
         disk, a server, a database or an interpreter.\n",
        violations.len()
    );
    for violation in &violations {
        message.push_str("  - ");
        message.push_str(&violation.describe());
        message.push('\n');
    }
    Err(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Shaped exactly like real output, including the workspace member's path
    /// suffix and the empty feature list that produces a trailing `| `.
    const PURE_TREE: &str = "\
batchalign-core v0.1.0 (/repo/crates/batchalign-core)
├── serde v1.0.229
│   └── serde_derive v1.0.229 (proc-macro)
└── thiserror v2.0.17
";

    fn at(path: &str) -> RepoRelativePath {
        RepoRelativePath(path.to_owned())
    }

    // -- the dependency half -------------------------------------------------

    #[test]
    fn a_pure_tree_reports_no_violations() {
        assert_eq!(classify_tree(PURE_TREE), Vec::new());
    }

    /// Asserted against the whole `Option` rather than unwrapping it: the
    /// workspace denies `expect()` even in test code, and the equality form says
    /// the same thing without a panic site.
    #[test]
    fn the_root_package_line_is_parsed_as_the_crate_itself() {
        assert_eq!(
            parse_tree_line("batchalign-core v0.1.0 (/repo/crates/batchalign-core)"),
            Some(PackageName("batchalign-core".to_owned()))
        );
    }

    /// Without `--no-dedupe` cargo marks a package it has already expanded, and
    /// that marker must not become part of the name.
    #[test]
    fn a_deduplicated_repeat_still_parses_as_its_package() {
        assert_eq!(
            parse_tree_line("└── serde v1.0.229 (*)"),
            Some(PackageName("serde".to_owned()))
        );
    }

    #[test]
    fn a_dependency_kind_header_is_not_a_package() {
        assert_eq!(parse_tree_line("[build-dependencies]"), None);
        assert_eq!(parse_tree_line(""), None);
    }

    #[test]
    fn an_axum_dependency_is_reported_as_an_http_server() {
        let tree = format!("{PURE_TREE}└── axum v0.8.7\n");
        assert_eq!(
            classify_tree(&tree),
            vec![Violation::ForbiddenPackage {
                package: PackageName("axum".to_owned()),
                capability: ForbiddenCapability::HttpServer,
            }]
        );
    }

    #[test]
    fn a_transitive_forbidden_crate_is_reported_even_when_nested() {
        let tree = format!("{PURE_TREE}└── some-wrapper v1.0.0\n    └── sqlx-core v0.8.7\n");
        assert_eq!(
            classify_tree(&tree),
            vec![Violation::ForbiddenPackage {
                package: PackageName("sqlx-core".to_owned()),
                capability: ForbiddenCapability::Database,
            }]
        );
    }

    /// The finding that reshaped this gate: cargo's feature unification makes
    /// core's tree show tokio with `full` and `mio` beneath it however narrowly
    /// core declares the dependency. Judging either one would fail the gate on a
    /// crate whose source is clean, so neither is a forbidden package.
    #[test]
    fn tokio_and_mio_in_the_tree_are_not_by_themselves_violations() {
        let tree = format!("{PURE_TREE}└── tokio v1.53.1\n    └── mio v1.2.2\n");
        assert_eq!(classify_tree(&tree), Vec::new());
    }

    /// A package can still appear more than once (cargo expands it once and
    /// marks later reaches), and a workspace member appears in its own tree. One
    /// violation per package, or the report buries its own finding.
    #[test]
    fn a_repeated_forbidden_crate_is_reported_once() {
        let tree = format!("{PURE_TREE}├── axum v0.8.7\n└── other v1.0.0\n    └── axum v0.8.7\n");
        assert_eq!(classify_tree(&tree).len(), 1);
    }

    // -- the source half -----------------------------------------------------

    #[test]
    fn a_tokio_spawn_in_core_source_is_a_violation() {
        let violations = scan_source_text(
            &at("crates/batchalign-core/src/runner.rs"),
            "pub fn go() {\n    tokio::spawn(work());\n}\n",
        );
        assert_eq!(
            violations,
            vec![Violation::ImpureSource {
                path: at("crates/batchalign-core/src/runner.rs"),
                line: LineNumber(2),
                marker: SourceMarker::AsyncRuntime,
                pattern: "tokio::spawn",
            }]
        );
    }

    /// Ambient in exactly the way the filesystem is: a core function that reads
    /// its own configuration out of the process environment cannot be tested by
    /// being given different inputs. Found while surveying `runtime_paths`,
    /// which the purity measurement calls 100% pure and which exists solely to
    /// read two environment variables.
    #[test]
    fn reading_the_process_environment_is_a_violation() {
        let violations = scan_source_text(
            &at("crates/batchalign-core/src/paths.rs"),
            "let dir = std::env::var(\"BATCHALIGN_CACHE_DIR\").ok();\n",
        );
        assert_eq!(violations.len(), 1);
        assert!(matches!(
            violations.first(),
            Some(Violation::ImpureSource {
                marker: SourceMarker::Environment,
                ..
            })
        ));
    }

    /// `env!` is resolved by the compiler and reads nothing at runtime, so it is
    /// not the same capability at all.
    #[test]
    fn the_compile_time_env_macro_is_allowed() {
        assert_eq!(
            scan_source_text(
                &at("crates/batchalign-core/src/version.rs"),
                "const VERSION: &str = env!(\"CARGO_PKG_VERSION\");\n",
            ),
            Vec::new()
        );
    }

    #[test]
    fn reading_a_file_in_core_source_is_a_violation() {
        let violations = scan_source_text(
            &at("crates/batchalign-core/src/load.rs"),
            "let text = std::fs::read_to_string(path)?;\n",
        );
        assert_eq!(violations.len(), 1);
        assert!(matches!(
            violations.first(),
            Some(Violation::ImpureSource {
                marker: SourceMarker::Filesystem,
                ..
            })
        ));
    }

    /// The trap this crate's own `lib.rs` walks straight into: its module doc
    /// names `std::fs` and `tokio::spawn` while the crate contains neither.
    #[test]
    fn a_marker_named_only_in_a_comment_is_not_a_violation() {
        let text = "\
//! Core must not use std::fs or tokio::spawn.
/// Never call Command::new here.
// std::net is out too.
pub fn pure() {}
";
        assert_eq!(
            scan_source_text(&at("crates/batchalign-core/src/lib.rs"), text),
            Vec::new()
        );
    }

    /// `tokio::sync` is the whole reason the tokio surfaces are separated: a
    /// task runner awaiting a channel is exactly what core is for.
    #[test]
    fn tokio_sync_and_async_trait_are_allowed_in_core_source() {
        let text = "\
use tokio::sync::mpsc;
#[async_trait::async_trait]
pub trait Dispatcher {
    async fn dispatch(&self, task: Task) -> Result<Output>;
}
";
        assert_eq!(
            scan_source_text(&at("crates/batchalign-core/src/dispatch.rs"), text),
            Vec::new()
        );
    }

    #[test]
    fn a_tokio_test_attribute_is_allowed() {
        let text = "    #[tokio::test]\n    async fn it_dispatches() {}\n";
        assert_eq!(
            scan_source_text(&at("crates/batchalign-core/src/dispatch.rs"), text),
            Vec::new()
        );
    }

    /// A line matching several patterns is one problem, not three, and a report
    /// that says otherwise inflates its own count.
    #[test]
    fn one_line_yields_at_most_one_violation() {
        let violations = scan_source_text(
            &at("crates/batchalign-core/src/x.rs"),
            "std::fs::write(std::process::id().to_string())?;\n",
        );
        assert_eq!(violations.len(), 1);
    }

    // -- the registries ------------------------------------------------------

    /// Both `ALL` arrays are hand-written, so nothing but a test stops a new
    /// variant from being declared and then never scanned for. The exhaustive
    /// `match` is the mechanism: adding a variant fails to compile here until
    /// its arm exists, and the arm is what asserts it was registered.
    #[test]
    fn every_forbidden_capability_is_in_all() {
        for capability in [
            ForbiddenCapability::ForeignExecutor,
            ForbiddenCapability::HttpServer,
            ForbiddenCapability::HttpClient,
            ForbiddenCapability::Database,
            ForbiddenCapability::PythonBridge,
        ] {
            match capability {
                ForbiddenCapability::ForeignExecutor
                | ForbiddenCapability::HttpServer
                | ForbiddenCapability::HttpClient
                | ForbiddenCapability::Database
                | ForbiddenCapability::PythonBridge => {}
            }
            assert!(
                ForbiddenCapability::ALL.contains(&capability),
                "{capability:?} is declared but never scanned for"
            );
            assert!(
                !capability.packages().is_empty(),
                "{capability:?} names no packages, so it can never fire"
            );
        }
    }

    #[test]
    fn every_source_marker_is_in_all() {
        for marker in [
            SourceMarker::AsyncRuntime,
            SourceMarker::Filesystem,
            SourceMarker::Subprocess,
            SourceMarker::Network,
            SourceMarker::Environment,
        ] {
            match marker {
                SourceMarker::AsyncRuntime
                | SourceMarker::Filesystem
                | SourceMarker::Subprocess
                | SourceMarker::Network
                | SourceMarker::Environment => {}
            }
            assert!(
                SourceMarker::ALL.contains(&marker),
                "{marker:?} is declared but never scanned for"
            );
            assert!(
                !marker.patterns().is_empty(),
                "{marker:?} has no patterns, so it can never fire"
            );
        }
    }

    // -- the message ---------------------------------------------------------

    /// The gate's whole value is that a stranger can act on the output without
    /// reading this module.
    #[test]
    fn the_dependency_message_names_the_crate_and_how_to_find_the_edge() {
        let message = Violation::ForbiddenPackage {
            package: PackageName("pyo3".to_owned()),
            capability: ForbiddenCapability::PythonBridge,
        }
        .describe();
        assert!(message.contains("pyo3"), "message was: {message}");
        assert!(
            message.contains("cargo tree -p batchalign-core -e normal -i pyo3"),
            "message was: {message}"
        );
    }

    #[test]
    fn the_source_message_names_the_file_line_and_pattern() {
        let message = Violation::ImpureSource {
            path: at("crates/batchalign-core/src/load.rs"),
            line: LineNumber(42),
            marker: SourceMarker::Filesystem,
            pattern: "std::fs",
        }
        .describe();
        assert!(
            message.contains("crates/batchalign-core/src/load.rs:42"),
            "message was: {message}"
        );
        assert!(message.contains("std::fs"), "message was: {message}");
    }
}
