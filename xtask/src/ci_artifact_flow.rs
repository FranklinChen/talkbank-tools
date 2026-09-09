//! Every job that runs this commit's CLI binary fetches it through one action.
//!
//! `build-cli` compiles `batchalign3` once and uploads it; several other jobs
//! run it. Uploading an artifact zips its files and drops the POSIX mode, so
//! the binary comes back at 644 and the executable bit has to be restored by
//! whoever downloads it. Five jobs each wrote out the download and the
//! `chmod`, and the pairing was held together by nothing but the habit of
//! copying the job above. A sixth job that copied only the first half would
//! fail deep inside a script as a permission error rather than as the missing
//! `chmod` it is.
//!
//! `.github/actions/cli-binary` owns both halves, so a caller cannot have one
//! without the other. This module is the other half of that change: building
//! the single route is worth nothing while the old route is still open.
//!
//! # Scope, and why it is not every workflow
//!
//! An artifact belongs to one workflow RUN, so only jobs in the same workflow
//! as the producing upload can download it. The scan therefore locates the
//! workflow that publishes `cli-binary` and applies its rules THERE, which is
//! both sound and what makes the rules affordable: inside that file an
//! artifact reference nobody can resolve is a red flag, while elsewhere it is
//! ordinary (the release workflow legitimately downloads by `pattern:` and by
//! a name interpolated from its build matrix).
//!
//! The producing upload itself is untouched: only downloads are judged, so
//! `build-cli` stays legal by construction rather than by sitting on an
//! allowlist somebody has to maintain.

use std::collections::BTreeMap;
use std::path::Path;

use serde::Deserialize;

/// The artifact name `build-cli` publishes and its consumers fetch.
const CLI_BINARY_ARTIFACT: &str = "cli-binary";

/// The composite action that is the only sanctioned way to fetch it.
const CLI_BINARY_ACTION: &str = "./.github/actions/cli-binary";

/// That action's own definition, which must keep doing both halves.
const CLI_BINARY_ACTION_FILE: &str = ".github/actions/cli-binary/action.yml";

/// Matched anywhere in a step's `uses`, so a third-party downloader and a
/// full-SHA pin are both recognised rather than only `actions/...@vN`.
const DOWNLOAD_ACTION_MARKER: &str = "download-artifact";

/// The same, for the producing side.
const UPLOAD_ACTION_MARKER: &str = "upload-artifact";

/// The binary the artifact carries.
const CLI_BINARY_NAME: &str = "batchalign3";

// ---------------------------------------------------------------------------
// The workflow, as much of it as this invariant needs
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct Workflow {
    #[serde(default)]
    jobs: BTreeMap<String, Job>,
}

#[derive(Debug, Deserialize)]
struct Job {
    /// Absent on a job that calls a reusable workflow rather than running steps.
    #[serde(default)]
    steps: Vec<Step>,
}

/// One composite action definition, read for the same fields as a step.
#[derive(Debug, Deserialize)]
struct ActionDefinition {
    runs: ActionRuns,
}

#[derive(Debug, Deserialize)]
struct ActionRuns {
    #[serde(default)]
    steps: Vec<Step>,
}

#[derive(Debug, Deserialize)]
struct Step {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    uses: Option<String>,
    #[serde(default)]
    run: Option<String>,
    #[serde(default)]
    with: BTreeMap<String, serde_yaml::Value>,
}

/// Which artifact a download step asks for.
///
/// `actions/download-artifact` takes EITHER a `name` or a `pattern`, and
/// either may interpolate an expression. Reading only `name` is how the first
/// version of this check could be walked straight past, so the reference is a
/// sum with a case for what cannot be resolved rather than an `Option<&str>`
/// that reads the same as "asks for nothing".
#[derive(Debug, PartialEq, Eq)]
enum ArtifactReference {
    /// A literal artifact name.
    Named(String),
    /// A glob over artifact names.
    Pattern(String),
    /// A name or pattern carrying a `${{ ... }}` expression, so what it
    /// resolves to is decided at run time and not here.
    Unresolvable(String),
    /// Neither key was given, which downloads every artifact in the run.
    EveryArtifact,
}

impl ArtifactReference {
    /// Whether this reference can reach the CLI binary artifact.
    fn reaches_cli_binary(&self) -> bool {
        match self {
            Self::Named(name) => name == CLI_BINARY_ARTIFACT,
            Self::Pattern(pattern) => glob_matches(pattern, CLI_BINARY_ARTIFACT),
            // Both by design: an unresolvable reference inside the producing
            // workflow could be the binary and nothing here can tell, and a
            // step naming neither key takes every artifact including this one.
            Self::Unresolvable(_) | Self::EveryArtifact => true,
        }
    }
}

/// Whether a `download-artifact` glob can match one artifact name.
///
/// Only `*` and `?` are handled, which is what the action's own matcher uses
/// for the shapes that appear in a workflow. Anything richer is conservative
/// in the right direction: an unrecognised metacharacter is left literal, and
/// a literal that does not match means the step is not asking for the binary.
fn glob_matches(pattern: &str, candidate: &str) -> bool {
    fn walk(pattern: &[u8], candidate: &[u8]) -> bool {
        match (pattern.first(), candidate.first()) {
            (None, None) => true,
            (None, Some(_)) => false,
            (Some(b'*'), _) => {
                // Consume nothing, or one more candidate byte and retry.
                walk(&pattern[1..], candidate)
                    || (!candidate.is_empty() && walk(pattern, &candidate[1..]))
            }
            (Some(_), None) => false,
            (Some(b'?'), Some(_)) => walk(&pattern[1..], &candidate[1..]),
            (Some(p), Some(c)) => p == c && walk(&pattern[1..], &candidate[1..]),
        }
    }
    walk(pattern.as_bytes(), candidate.as_bytes())
}

impl Step {
    /// How this step is named in a failure message: its own `name`, else the
    /// action it calls, else its script's first line.
    fn describe(&self) -> String {
        if let Some(name) = &self.name {
            return name.clone();
        }
        if let Some(uses) = &self.uses {
            return format!("uses: {uses}");
        }
        match &self.run {
            Some(script) => match script.lines().next() {
                Some(first) => format!("run: {}", first.trim()),
                None => "an empty run step".into(),
            },
            None => "an unnamed step".into(),
        }
    }

    /// Whether this step calls an artifact download action, from any owner and
    /// pinned any way.
    fn is_artifact_download(&self) -> bool {
        self.uses
            .as_deref()
            .is_some_and(|uses| uses.contains(DOWNLOAD_ACTION_MARKER))
    }

    /// The artifact a download step asks for.
    fn artifact_reference(&self) -> ArtifactReference {
        let read = |key: &str| self.with.get(key).and_then(serde_yaml::Value::as_str);
        match (read("name"), read("pattern")) {
            (Some(raw), _) | (None, Some(raw)) if raw.contains("${{") => {
                ArtifactReference::Unresolvable(raw.to_owned())
            }
            (Some(name), _) => ArtifactReference::Named(name.to_owned()),
            (None, Some(pattern)) => ArtifactReference::Pattern(pattern.to_owned()),
            (None, None) => ArtifactReference::EveryArtifact,
        }
    }

    /// The artifact an upload step publishes, if it is an upload at all.
    fn published_artifact(&self) -> Option<ArtifactReference> {
        let uses = self.uses.as_deref()?;
        if !uses.contains(UPLOAD_ACTION_MARKER) {
            return None;
        }
        Some(self.artifact_reference())
    }

    /// Whether this step restores the CLI binary's executable bit by hand.
    ///
    /// Any mode, and `install -m` too, because the first version matched the
    /// literal `chmod +x` and `chmod 755` walked past it. Scoped to the
    /// producing workflow by the caller, so an unrelated script elsewhere that
    /// happens to chmod something named for the binary is not swept in.
    fn restores_executable_bit(&self) -> bool {
        let Some(script) = &self.run else {
            return false;
        };
        script.contains(CLI_BINARY_NAME)
            && (script.contains("chmod") || script.contains("install -m"))
    }
}

// ---------------------------------------------------------------------------
// Findings
// ---------------------------------------------------------------------------

/// A route to the CLI binary that goes around the composite action.
#[derive(Debug, PartialEq, Eq)]
enum Bypass {
    /// The job fetched the artifact itself, so nothing makes it executable.
    RawDownload {
        job: String,
        step: String,
        reference: ArtifactReference,
    },
    /// The job restored the executable bit by hand, which is the action's job.
    HandRestoredBit { job: String, step: String },
}

impl Bypass {
    fn message(&self) -> String {
        match self {
            Self::RawDownload {
                job,
                step,
                reference,
            } => {
                let asked = match reference {
                    ArtifactReference::Named(name) => format!("names the `{name}` artifact"),
                    ArtifactReference::Pattern(pattern) => {
                        format!(
                            "uses the pattern `{pattern}`, which matches `{CLI_BINARY_ARTIFACT}`"
                        )
                    }
                    ArtifactReference::Unresolvable(raw) => format!(
                        "asks for `{raw}`, which this check cannot resolve, and inside this \
                         workflow an unresolvable reference could be the CLI binary"
                    ),
                    ArtifactReference::EveryArtifact => {
                        "names no artifact, so it downloads every one in the run".to_owned()
                    }
                };
                format!(
                    "job `{job}`, step `{step}` downloads an artifact directly: it {asked}. The \
                     upload drops the executable bit, so a raw download yields a binary that \
                     cannot run. Use `uses: {CLI_BINARY_ACTION}` with a `path:` instead."
                )
            }
            Self::HandRestoredBit { job, step } => format!(
                "job `{job}`, step `{step}` restores the CLI binary's executable bit by hand. \
                 `{CLI_BINARY_ACTION}` already does that as part of fetching it; delete the step."
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// The scan
// ---------------------------------------------------------------------------

/// Parse one workflow, or say why it could not be read.
fn parse_workflow(text: &str) -> std::result::Result<Workflow, String> {
    serde_yaml::from_str(text).map_err(|err| format!("cannot parse as a workflow: {err}"))
}

/// Whether this workflow publishes the CLI binary artifact.
fn publishes_cli_binary(workflow: &Workflow) -> bool {
    workflow.jobs.values().any(|job| {
        job.steps.iter().any(|step| {
            step.published_artifact()
                == Some(ArtifactReference::Named(CLI_BINARY_ARTIFACT.to_owned()))
        })
    })
}

/// Every bypass in the workflow that publishes the CLI binary, in file order.
fn scan_producing_workflow(workflow: &Workflow) -> Vec<Bypass> {
    let mut bypasses = Vec::new();
    for (job_name, job) in &workflow.jobs {
        for step in &job.steps {
            if step.is_artifact_download() {
                let reference = step.artifact_reference();
                if reference.reaches_cli_binary() {
                    bypasses.push(Bypass::RawDownload {
                        job: job_name.clone(),
                        step: step.describe(),
                        reference,
                    });
                }
            }
            if step.restores_executable_bit() {
                bypasses.push(Bypass::HandRestoredBit {
                    job: job_name.clone(),
                    step: step.describe(),
                });
            }
        }
    }
    bypasses
}

/// The action must keep doing both halves, or every workflow passes this check
/// while shipping a binary that cannot run.
///
/// Read as a DEFINITION, not as text. The first version searched the file for
/// substrings, so deleting the `chmod` step while leaving a sentence about it
/// in the description reported clean, which is the exact failure this module's
/// own docs call the worst kind of guard.
fn check_action_definition(root: &Path) -> std::result::Result<(), String> {
    let path = root.join(CLI_BINARY_ACTION_FILE);
    let text = std::fs::read_to_string(&path)
        .map_err(|err| format!("cannot read {CLI_BINARY_ACTION_FILE}: {err}"))?;
    let definition: ActionDefinition = serde_yaml::from_str(&text)
        .map_err(|err| format!("cannot parse {CLI_BINARY_ACTION_FILE} as an action: {err}"))?;

    let mut missing = Vec::new();
    let downloads = definition.runs.steps.iter().any(|step| {
        step.is_artifact_download()
            && step.artifact_reference() == ArtifactReference::Named(CLI_BINARY_ARTIFACT.to_owned())
    });
    if !downloads {
        missing.push(format!(
            "it must have a step that downloads the `{CLI_BINARY_ARTIFACT}` artifact"
        ));
    }
    if !definition
        .runs
        .steps
        .iter()
        .any(Step::restores_executable_bit)
    {
        missing.push(
            "it must have a step that restores the executable bit the upload dropped".to_owned(),
        );
    }

    if missing.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "{CLI_BINARY_ACTION_FILE} no longer does what the workflows rely on: {}",
            missing.join("; ")
        ))
    }
}

/// Public entry point: the composite action is the only route to the binary.
pub fn check(root: &Path) -> std::result::Result<(), String> {
    let mut failures = Vec::new();

    if let Err(msg) = check_action_definition(root) {
        failures.push(msg);
    }

    let workflows = root.join(".github/workflows");
    let entries = std::fs::read_dir(&workflows)
        .map_err(|err| format!("cannot read {}: {err}", workflows.display()))?;

    // Sorted, so a failure list is stable between runs and between machines.
    let mut paths: Vec<_> = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|err| format!("cannot read a workflow entry: {err}"))?;
        let path = entry.path();
        if path
            .extension()
            .is_some_and(|ext| ext == "yml" || ext == "yaml")
        {
            paths.push(path);
        }
    }
    paths.sort();

    let mut producers = Vec::new();
    for path in paths {
        let name = path.file_name().map_or_else(
            || path.display().to_string(),
            |n| n.to_string_lossy().into(),
        );
        let text = std::fs::read_to_string(&path)
            .map_err(|err| format!("cannot read workflow {name}: {err}"))?;
        match parse_workflow(&text) {
            Err(msg) => failures.push(format!(".github/workflows/{name}: {msg}")),
            Ok(workflow) => {
                if publishes_cli_binary(&workflow) {
                    producers.push(name.clone());
                    for bypass in scan_producing_workflow(&workflow) {
                        failures.push(format!(".github/workflows/{name}: {}", bypass.message()));
                    }
                }
            }
        }
    }

    // A check whose subject has disappeared reports clean, which is worse than
    // no check. Say so instead.
    match producers.len() {
        1 => {}
        0 => failures.push(format!(
            "no workflow publishes a `{CLI_BINARY_ARTIFACT}` artifact, so this check examined \
             nothing; either the artifact was renamed or the producing job is gone"
        )),
        _ => failures.push(format!(
            "more than one workflow publishes a `{CLI_BINARY_ARTIFACT}` artifact ({}), so which \
             binary a consuming job receives depends on which workflow it runs in",
            producers.join(", ")
        )),
    }

    if failures.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "CLI binary artifact flow check failed:\n- {}",
            failures.join("\n- ")
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::{ArtifactReference, Bypass, glob_matches, parse_workflow, scan_producing_workflow};

    /// Parse a workflow and scan it, for a test that has already decided the
    /// workflow is the producing one.
    fn scan(text: &str) -> Vec<Bypass> {
        let workflow = parse_workflow(text).unwrap_or_else(|err| panic!("{err}"));
        scan_producing_workflow(&workflow)
    }

    /// The route the composite action replaced. If this passes, the old route
    /// is still open and building the action bought nothing.
    #[test]
    fn a_job_that_downloads_the_artifact_by_name_is_refused() {
        let bypasses = scan(
            r#"
jobs:
  dashboard-smoke:
    steps:
      - uses: actions/download-artifact@v8
        with:
          name: cli-binary
          path: target/debug/
"#,
        );
        assert_eq!(
            bypasses,
            vec![Bypass::RawDownload {
                job: "dashboard-smoke".into(),
                step: "uses: actions/download-artifact@v8".into(),
                reference: ArtifactReference::Named("cli-binary".into()),
            }]
        );
    }

    /// The hole a review found: the action takes a `pattern` INSTEAD of a
    /// `name`, and the first version of this check read only `name`, so this
    /// walked straight past it. The release workflow already uses `pattern`,
    /// so this is a shape somebody here writes, not a hypothetical.
    #[test]
    fn a_job_that_downloads_by_a_matching_pattern_is_refused() {
        let bypasses = scan(
            r#"
jobs:
  sneaky:
    steps:
      - uses: actions/download-artifact@v8
        with:
          pattern: cli-*
          merge-multiple: true
"#,
        );
        assert_eq!(bypasses.len(), 1, "{bypasses:?}");
    }

    /// A reference nobody can resolve is refused rather than passed, because
    /// inside the producing workflow it could be the binary and nothing here
    /// can tell. Three buckets, and this is the middle one.
    #[test]
    fn a_job_that_downloads_an_interpolated_name_is_refused() {
        let bypasses = scan(
            r#"
jobs:
  matrixed:
    steps:
      - uses: actions/download-artifact@v8
        with:
          name: ${{ matrix.artifact }}
"#,
        );
        assert_eq!(bypasses.len(), 1, "{bypasses:?}");
    }

    /// Naming neither key downloads every artifact in the run, this one
    /// included.
    #[test]
    fn a_job_that_downloads_everything_is_refused() {
        let bypasses = scan(
            r#"
jobs:
  greedy:
    steps:
      - uses: actions/download-artifact@v8
"#,
        );
        assert_eq!(bypasses.len(), 1, "{bypasses:?}");
    }

    /// A third-party downloader is still a downloader.
    #[test]
    fn a_third_party_download_action_is_refused() {
        let bypasses = scan(
            r#"
jobs:
  elsewhere:
    steps:
      - uses: dawidd6/action-download-artifact@v6
        with:
          name: cli-binary
"#,
        );
        assert_eq!(bypasses.len(), 1, "{bypasses:?}");
    }

    /// Any mode, not the one spelling. `chmod 755` walked past the first
    /// version of this rule.
    #[test]
    fn restoring_the_bit_by_hand_is_refused_whatever_the_mode() {
        for script in [
            "chmod +x target/debug/batchalign3",
            "chmod 755 target/debug/batchalign3",
            "install -m 0755 target/debug/batchalign3 /usr/local/bin/batchalign3",
        ] {
            let workflow = format!(
                r#"
jobs:
  rust-integration:
    steps:
      - name: Make CLI binary executable
        run: {script}
"#
            );
            let bypasses = scan(&workflow);
            assert_eq!(
                bypasses,
                vec![Bypass::HandRestoredBit {
                    job: "rust-integration".into(),
                    step: "Make CLI binary executable".into(),
                }],
                "not refused: {script}"
            );
        }
    }

    /// The near miss that decides whether this check is worth having. The job
    /// that PUBLISHES the artifact names it too, and flagging that would make
    /// the check unusable, so it distinguishes producer from consumer rather
    /// than searching for the artifact's name.
    #[test]
    fn publishing_the_artifact_is_not_a_bypass() {
        let bypasses = scan(
            r#"
jobs:
  build-cli:
    steps:
      - name: Build development CLI binary
        run: cargo build -p batchalign
      - uses: actions/upload-artifact@v7
        with:
          name: cli-binary
          path: target/debug/batchalign3
          if-no-files-found: error
"#,
        );
        assert!(bypasses.is_empty(), "{bypasses:?}");
    }

    /// Downloading a DIFFERENT artifact by a literal name stays legal.
    #[test]
    fn downloading_another_artifact_by_name_is_not_a_bypass() {
        let bypasses = scan(
            r#"
jobs:
  typecheck:
    steps:
      - uses: actions/download-artifact@v8
        with:
          name: wheel
          path: dist/
      - uses: ./.github/actions/cli-binary
        with:
          path: target/debug
"#,
        );
        assert!(bypasses.is_empty(), "{bypasses:?}");
    }

    /// A pattern that cannot match the binary stays legal, so the glob is not
    /// just returning true.
    #[test]
    fn a_pattern_that_cannot_match_is_not_a_bypass() {
        let bypasses = scan(
            r#"
jobs:
  release:
    steps:
      - uses: actions/download-artifact@v8
        with:
          pattern: wheel-*
          merge-multiple: true
"#,
        );
        assert!(bypasses.is_empty(), "{bypasses:?}");
    }

    /// A job that calls a reusable workflow has no `steps` key at all.
    #[test]
    fn a_job_without_steps_is_read_without_failing() {
        let bypasses = scan(
            r#"
jobs:
  call-release:
    uses: ./.github/workflows/batchalign-release.yml
"#,
        );
        assert!(bypasses.is_empty(), "{bypasses:?}");
    }

    /// The glob decides which downloads are refused, so its own boundaries are
    /// worth pinning in both directions.
    #[test]
    fn the_glob_matches_only_what_it_should() {
        assert!(glob_matches("cli-binary", "cli-binary"));
        assert!(glob_matches("cli-*", "cli-binary"));
        assert!(glob_matches("*", "cli-binary"));
        assert!(glob_matches("cli-binar?", "cli-binary"));
        assert!(glob_matches("*binary", "cli-binary"));
        assert!(!glob_matches("wheel-*", "cli-binary"));
        assert!(!glob_matches("cli-binaries", "cli-binary"));
        assert!(!glob_matches("cli-binar", "cli-binary"));
    }
}
