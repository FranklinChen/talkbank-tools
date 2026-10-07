//! Write server job results to the local filesystem.
//!
//! After the CLI polls a completed job, each [`FileResult`] must be written to
//! the correct output path. This module handles:
//!
//! - **Path resolution**: the discovery producer admits one source-bound plan
//!   covering command-owned primary artifacts and sidecars. Unknown results
//!   cannot acquire a destination through a filename fallback.
//! - **Path traversal protection**: the resolved output path is checked against
//!   the canonicalized output directory so a malicious server cannot write
//!   outside the intended tree (e.g. `../../../etc/passwd`).
//! - **Parent directory creation**: intermediate directories are created
//!   automatically so callers do not need to pre-create nested output trees.
//!
//! Roots and already-rooted destinations are separate types. Admission resolves
//! existing symlinks without creating directories; writing rechecks containment
//! against the live filesystem. Relative `-o` is resolved against the process
//! directory once, never joined onto itself. Without `-o`, each source retains
//! its own parent rather than borrowing the first input's directory.

use std::collections::HashMap;
use std::path::{Component, Path, PathBuf};

use crate::api::{ContentType, FileResult, ResultContent};
mod binary;
pub use binary::BinaryResultRefusal;

#[cfg(test)]
use crate::ReleasedCommand;
use crate::cli::discover::PlannedServerInput;
use crate::cli::error::CliError;

/// Exact result identities and write capabilities admitted by discovery.
///
/// No public constructor, mutable access or default can fabricate this proof.
/// The same command naming owner drives runner artifacts and this plan.
#[derive(Debug)]
pub struct ResultDestinations {
    artifacts: HashMap<String, ResultDestination>,
}

#[derive(Debug)]
struct ResultDestination {
    path: PlannedOutputPath,
    root: OutputRoot,
    content_type: ContentType,
}

impl ResultDestinations {
    /// Common directory for progress summaries, derived from admitted roots.
    /// This projection grants no write permission.
    pub fn reporting_directory(&self) -> Result<PathBuf, CliError> {
        let mut roots = self
            .artifacts
            .values()
            .map(|artifact| artifact.root.as_path());
        let mut common = roots
            .next()
            .ok_or_else(|| {
                CliError::InvalidArgument("cannot summarize an empty destination plan".into())
            })?
            .to_path_buf();
        for root in roots {
            while !root.starts_with(&common) {
                if !common.pop() {
                    return Err(CliError::InvalidArgument(
                        "admitted output roots have no common directory".into(),
                    ));
                }
            }
        }
        Ok(common)
    }

    pub(super) fn admit(
        options: &crate::options::CommandOptions,
        inputs: &[PlannedServerInput],
        out_dir: Option<&Path>,
    ) -> Result<Self, CliError> {
        let explicit_root = out_dir.map(OutputRoot::admit).transpose()?;
        let mut artifacts = HashMap::new();
        let mut destinations: HashMap<PathBuf, PathBuf> = HashMap::new();
        let mut names: HashMap<String, PathBuf> = HashMap::new();
        let sources: std::collections::HashSet<_> = inputs
            .iter()
            .map(|input| input.source().to_path_buf())
            .collect();
        for input in inputs {
            let parent = input.output_anchor().parent().ok_or_else(|| {
                CliError::InvalidArgument("planned output anchor has no parent".into())
            })?;
            // With no -o, each source retains its own admitted parent; a
            // mixed-root in-place job is not confined to the first input root.
            let root = match &explicit_root {
                Some(root) => root.clone(),
                None => OutputRoot::admit(input.source().parent().ok_or_else(|| {
                    CliError::InvalidArgument("planned source has no parent".into())
                })?)?,
            };
            let outputs = crate::recipe_runner::runtime::planned_output_artifacts(
                options.command(),
                options,
                &input.server_name().into(),
            )
            .map_err(|error| CliError::InvalidArgument(error.to_string()))?;
            for output in outputs {
                let name = output.display_path.to_string();
                let basename = Path::new(&name).file_name().ok_or_else(|| {
                    CliError::InvalidArgument("planned result has no filename".into())
                })?;
                let path = PlannedOutputPath::already_planned(&parent.join(basename))?;
                let identity = path.checked_under(&root)?;
                if output.content_type.is_binary() {
                    if sources.contains(&identity) {
                        return Err(CliError::InvalidArgument(
                            "an audio export would replace a submitted source".into(),
                        ));
                    }
                    match std::fs::symlink_metadata(&path.0) {
                        Ok(_) => {
                            return Err(CliError::BinaryResult {
                                filename: output.display_path.clone(),
                                refusal: BinaryResultRefusal::Existing(path.0.clone()),
                            });
                        }
                        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                        Err(error) => return Err(error.into()),
                    }
                }
                if let Some(first) =
                    destinations.insert(identity.clone(), input.source().to_path_buf())
                {
                    return Err(CliError::OutputCollision {
                        destination: identity,
                        first,
                        second: input.source().to_path_buf(),
                    });
                }
                if let Some(first) = names.insert(name.clone(), input.source().to_path_buf()) {
                    return Err(CliError::InputNameCollision {
                        name,
                        first,
                        second: input.source().to_path_buf(),
                    });
                }
                artifacts.insert(
                    name,
                    ResultDestination {
                        path,
                        root: root.clone(),
                        content_type: output.content_type,
                    },
                );
            }
        }
        Ok(Self { artifacts })
    }
}

/// Absolute output anchor with resolved ancestors and the declared leaf intact.
/// The runner applies its artifact basename under this anchor's parent. Following
/// a final-file symlink here would silently move that parent before submission.
pub(super) fn absolute_output_anchor(path: &Path) -> Result<PathBuf, CliError> {
    let planned = PlannedOutputPath::already_planned(path)?;
    let parent = planned
        .0
        .parent()
        .ok_or_else(|| CliError::InvalidArgument("output anchor has no parent".into()))?;
    let leaf = planned
        .0
        .file_name()
        .ok_or_else(|| CliError::InvalidArgument("output anchor has no filename".into()))?;
    Ok(resolved_without_creating(parent).join(leaf))
}

/// An absolute output authority with existing ancestors symlink-resolved.
/// Admission creates nothing; preparation materializes the directory for writing.
#[derive(Debug, Clone)]
struct OutputRoot(PathBuf);

impl OutputRoot {
    fn admit(out_dir: &Path) -> Result<Self, CliError> {
        let planned = PlannedOutputPath::already_planned(out_dir)?;
        Ok(Self(resolved_without_creating(&planned.0)))
    }
    /// Create the output directory if absent, then canonicalize it.
    ///
    /// Canonicalizing matters beyond tidiness: the containment check below
    /// compares against a canonicalized parent, and on macOS a temporary
    /// directory reached through `/var` canonicalizes to `/private/var`, so
    /// both sides must have been through the same resolution.
    fn prepare(out_dir: &Path) -> Result<Self, CliError> {
        std::fs::create_dir_all(out_dir).map_err(|e| {
            CliError::Io(std::io::Error::new(
                e.kind(),
                format!("cannot create output directory {}: {e}", out_dir.display()),
            ))
        })?;
        let canonical = std::fs::canonicalize(out_dir).map_err(|e| {
            CliError::Io(std::io::Error::new(
                e.kind(),
                format!("cannot resolve output directory {}: {e}", out_dir.display()),
            ))
        })?;
        Ok(Self(canonical))
    }

    fn as_path(&self) -> &Path {
        &self.0
    }
}

/// Where a single result file is to be written: absolute, and already rooted
/// at an [`OutputRoot`].
///
/// The inner path is private and there is no accessor that composes, so the
/// double join that produced `B/B` cannot be written. Its constructor
/// accept only a path the discovery producer already rooted. Server display
/// names cannot construct one at the write boundary.
#[derive(Debug, Clone)]
struct PlannedOutputPath(PathBuf);

impl PlannedOutputPath {
    /// Adopt a path that the discovery pass already rooted at the output
    /// directory (`out_dir.join(rel)`).
    ///
    /// Such a path is relative exactly when `-o` was given as a relative
    /// path, in which case it is relative to the PROCESS's current directory,
    /// never to the output root: resolving it against the root is what
    /// duplicated the directory. Containment is not assumed here, it is
    /// enforced by [`PlannedOutputPath::verified_under`].
    fn already_planned(path: &Path) -> Result<Self, CliError> {
        let absolute = if path.is_absolute() {
            path.to_path_buf()
        } else {
            let cwd = std::env::current_dir().map_err(|e| {
                CliError::Io(std::io::Error::new(
                    e.kind(),
                    format!(
                        "cannot read current directory to place {}: {e}",
                        path.display()
                    ),
                ))
            })?;
            cwd.join(path)
        };
        Ok(Self(lexically_normalized(&absolute)))
    }

    /// Check containment without changing the filesystem.
    fn checked_under(&self, root: &OutputRoot) -> Result<PathBuf, CliError> {
        let resolved = resolved_without_creating(&self.0);
        if !resolved.starts_with(root.as_path()) {
            return Err(CliError::PathTraversal(
                self.0.to_string_lossy().to_string(),
            ));
        }
        Ok(resolved)
    }

    /// Prove the destination sits inside `root`, create its parent, and
    /// return the path to write.
    ///
    /// Checked twice on purpose. The first check runs BEFORE any directory is
    /// created, so a hostile path never leaves a directory behind on its way
    /// to being refused; the second runs after the parent exists, because
    /// only the filesystem can settle a symlink planted between the two.
    fn verified_under(self, root: &OutputRoot) -> Result<PathBuf, CliError> {
        let traversal = || CliError::PathTraversal(self.0.to_string_lossy().to_string());

        let resolved = self.checked_under(root)?;

        let parent = resolved.parent().ok_or_else(traversal)?;
        std::fs::create_dir_all(parent).map_err(|e| {
            CliError::Io(std::io::Error::new(
                e.kind(),
                format!("cannot create output directory {}: {e}", parent.display()),
            ))
        })?;
        let canonical_parent = std::fs::canonicalize(parent).map_err(|e| {
            CliError::Io(std::io::Error::new(
                e.kind(),
                format!("cannot resolve output directory {}: {e}", parent.display()),
            ))
        })?;
        if !canonical_parent.starts_with(root.as_path()) {
            return Err(traversal());
        }

        let file_name = resolved.file_name().ok_or_else(traversal)?;
        Ok(canonical_parent.join(file_name))
    }
}

/// Resolve symlinks as far as the filesystem already reaches, creating
/// nothing.
///
/// Walks up to the deepest ancestor that exists, canonicalizes it, then
/// re-attaches the components that do not exist yet. This is what makes the
/// pre-creation containment check trustworthy: on macOS a temporary directory
/// handed in as `/var/folders/...` canonicalizes to `/private/var/folders/...`,
/// and comparing the unresolved form against a canonical root would refuse
/// every legitimate write. Expects `path` to be lexically normalized already,
/// so no `..` remains to be reinterpreted after a symlink is followed.
fn resolved_without_creating(path: &Path) -> PathBuf {
    let mut unresolved: Vec<std::ffi::OsString> = Vec::new();
    let mut cursor = path.to_path_buf();
    loop {
        if let Ok(canonical) = std::fs::canonicalize(&cursor) {
            let mut resolved = canonical;
            for component in unresolved.iter().rev() {
                resolved.push(component);
            }
            return resolved;
        }
        let Some(name) = cursor.file_name().map(|n| n.to_os_string()) else {
            // Reached a root that does not canonicalize: nothing to resolve.
            return path.to_path_buf();
        };
        unresolved.push(name);
        if !cursor.pop() {
            return path.to_path_buf();
        }
    }
}

/// Resolve `.` and `..` without consulting the filesystem.
///
/// Purely lexical so it can run BEFORE any directory is created. Symlink
/// escapes survive this and are caught by the canonical check afterwards.
fn lexically_normalized(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// Write a single file result to the output directory.
///
/// Only the producer-admitted exact name and content type permit a write.
///
/// Returns `Ok(true)` on success, `Ok(false)` if the result had an error,
/// or `Err` on I/O failure.
pub fn write_result(
    result: &FileResult,
    destinations: &ResultDestinations,
) -> Result<bool, CliError> {
    // Server-side error → skip
    if result.error.is_some() {
        return Ok(false);
    }

    let ResultContent::Text(content) = &result.content else {
        return Err(CliError::ResultTypeMismatch(result.filename.clone()));
    };
    if result.content_type.is_binary() {
        return Err(CliError::ResultTypeMismatch(result.filename.clone()));
    }

    let planned = destinations
        .artifacts
        .get(result.filename.as_ref())
        .ok_or_else(|| CliError::UnplannedResult(result.filename.clone()))?;
    if result.content_type != planned.content_type {
        return Err(CliError::ResultTypeMismatch(result.filename.clone()));
    }
    planned.path.checked_under(&planned.root)?;
    let root = OutputRoot::prepare(planned.root.as_path())?;
    let destination = planned.path.clone().verified_under(&root)?;
    std::fs::write(&destination, content)?;
    Ok(true)
}

/// Deliver a managed result through its admitted destination. Binary payloads
/// retain streamed identity verification and new-only publication; they never
/// enter the text writer or receive authority from their wire filename alone.
pub async fn write_remote_result(
    result: &FileResult,
    destinations: &ResultDestinations,
    client: &crate::cli::client::BatchalignClient,
    server_url: &str,
    job_id: &crate::api::JobId,
) -> Result<bool, CliError> {
    if result.error.is_some() {
        return Ok(false);
    }
    match &result.content {
        ResultContent::Text(_) => write_result(result, destinations),
        ResultContent::Binary(descriptor) => {
            let plan = binary::BinaryWritePlan::admit(result, descriptor, destinations)?;
            if plan.existing_matches().await? {
                return Ok(true);
            }
            let response = client
                .get_binary_result(server_url, job_id, &result.filename)
                .await?;
            plan.receive(response).await?.publish()?;
            Ok(true)
        }
    }
}

#[cfg(test)]
mod tests;
