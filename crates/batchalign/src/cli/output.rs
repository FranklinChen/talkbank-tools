//! Write server job results to the local filesystem.
//!
//! After the CLI polls a completed job, each [`FileResult`] must be written to
//! the correct output path. This module handles:
//!
//! - **Path resolution**: a pre-built `result_map` (server filename to local
//!   output path) provides exact lookup; a fallback joins the filename with the
//!   output directory and, for transcribe jobs that rename extensions, ensures
//!   the output gets a `.cha` suffix.
//! - **Path traversal protection**: the resolved output path is checked against
//!   the canonicalized output directory so a malicious server cannot write
//!   outside the intended tree (e.g. `../../../etc/passwd`).
//! - **Parent directory creation**: intermediate directories are created
//!   automatically so callers do not need to pre-create nested output trees.
//!
//! # Two kinds of path, two types
//!
//! Every path here is one of exactly two kinds, and conflating them produced
//! two field failures at once (external user report, 2026-08-11: `-o B` wrote
//! its results into `B/B`):
//!
//! - An [`OutputRoot`] is the job's output directory, canonical and absolute.
//! - A [`PlannedOutputPath`] is where one result file goes. It is absolute,
//!   and it is *already rooted* at the [`OutputRoot`].
//!
//! The old code held both as bare `PathBuf` and recovered the distinction at
//! runtime with `if out_path.is_absolute()`, which is a guess, not a fact. A
//! path built as `out_dir.join(rel)` from a RELATIVE `-o` is relative, so that
//! branch rooted it a second time and produced `B/B/session.cha`; an absolute
//! `-o` took the other branch and was correct, which is why every test in this
//! file (all using an absolute `tempfile::tempdir()`) passed. The same guess
//! also broke containment: a not-yet-existing relative output directory failed
//! to canonicalize and fell back to itself, so a relative prefix was compared
//! against an absolute parent and every write was rejected as a traversal.
//!
//! Neither mistake is expressible now: a [`PlannedOutputPath`] cannot be joined
//! onto anything, and an [`OutputRoot`] exists only after the directory does.

use std::collections::HashMap;
use std::path::{Component, Path, PathBuf};

use crate::api::{ContentType, FileResult};

use crate::cli::error::CliError;

/// The job's output directory: absolute, canonical, and known to exist.
///
/// Construction is the proof: [`OutputRoot::prepare`] is the only way to get
/// one, and it creates the directory before canonicalizing it. A caller
/// holding an `OutputRoot` therefore cannot be holding a relative path, which
/// is what made the containment check compare incomparable kinds.
struct OutputRoot(PathBuf);

impl OutputRoot {
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
/// double join that produced `B/B` cannot be written. The two constructors
/// name the two situations the old code could not tell apart:
/// [`PlannedOutputPath::already_planned`] for a path the discovery pass
/// already rooted, and [`PlannedOutputPath::under`] for a bare server-side
/// display name that still has to be placed.
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

    /// Place a server-side display name (`"sample.cha"`, `"PWA/TYO_a1.cha"`)
    /// directly under the output root.
    ///
    /// An absolute or upward-reaching name from the server survives the join
    /// unchanged or escapes the root; both are caught by
    /// [`PlannedOutputPath::verified_under`] rather than silently written.
    fn under(root: &OutputRoot, name: &Path) -> Self {
        Self(lexically_normalized(&root.as_path().join(name)))
    }

    /// Rewrite the extension to `.cha` unless the result is already CHAT or a
    /// CSV sidecar.
    ///
    /// POLICY, not an invariant: `transcribe` is handed media (`audio.mp3`)
    /// and returns a transcript, so the output file has to be renamed. Kept as
    /// a test because no type can state which commands rename.
    fn with_chat_extension(self) -> Self {
        let ext = self
            .0
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();
        if ext == "cha" || ext == "csv" {
            self
        } else {
            Self(self.0.with_extension("cha"))
        }
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

        let resolved = resolved_without_creating(&self.0);
        if !resolved.starts_with(root.as_path()) {
            return Err(traversal());
        }

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
/// Uses `result_map` (server_name -> output_path) for exact lookup.
/// Falls back to placing the file directly under `out_dir` if not in the map.
///
/// Returns `Ok(true)` on success, `Ok(false)` if the result had an error,
/// or `Err` on I/O failure.
pub fn write_result(
    result: &FileResult,
    result_map: &HashMap<String, PathBuf>,
    out_dir: &Path,
) -> Result<bool, CliError> {
    // Server-side error → skip
    if result.error.is_some() {
        return Ok(false);
    }

    let root = OutputRoot::prepare(out_dir)?;

    let planned = if result.content_type != ContentType::Chat {
        // Non-CHAT output (e.g. CSV from opensmile), use server filename directly
        PlannedOutputPath::under(&root, Path::new(&*result.filename))
    } else {
        resolve_output_path(&result.filename, result_map, &root)?
    };

    let destination = planned.verified_under(&root)?;
    std::fs::write(&destination, &result.content)?;
    Ok(true)
}

/// Map a result filename back to the correct output path.
fn resolve_output_path(
    result_filename: &str,
    result_map: &HashMap<String, PathBuf>,
    root: &OutputRoot,
) -> Result<PlannedOutputPath, CliError> {
    if let Some(path) = result_map.get(result_filename) {
        return PlannedOutputPath::already_planned(path);
    }
    // Fallback: place under the output root, renaming media extensions to .cha
    Ok(PlannedOutputPath::under(root, Path::new(result_filename)).with_chat_extension())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::FileProvenance;

    /// Resolve against a real (absolute) root and report where the file would
    /// land, so the assertions below read as plain paths.
    fn resolved(filename: &str, result_map: &HashMap<String, PathBuf>, root_dir: &Path) -> PathBuf {
        let root = OutputRoot::prepare(root_dir).unwrap();
        resolve_output_path(filename, result_map, &root)
            .unwrap()
            .verified_under(&root)
            .unwrap()
    }

    #[test]
    fn resolve_output_uses_map() {
        let dir = tempfile::tempdir().unwrap();
        let root = std::fs::canonicalize(dir.path()).unwrap();
        let mut map = HashMap::new();
        map.insert("test.cha".to_string(), root.join("test.cha"));

        assert_eq!(
            resolved("test.cha", &map, dir.path()),
            root.join("test.cha")
        );
    }

    #[test]
    fn resolve_output_fallback() {
        let dir = tempfile::tempdir().unwrap();
        let root = std::fs::canonicalize(dir.path()).unwrap();

        assert_eq!(
            resolved("test.cha", &HashMap::new(), dir.path()),
            root.join("test.cha")
        );
    }

    /// The field failure of 2026-08-11, at unit scale: a map value that was
    /// built from a RELATIVE `-o` must land where it says, not one level
    /// deeper. See `tests/relative_output_dir.rs` for the same contract
    /// exercised through real discovery.
    #[test]
    fn relative_planned_path_is_not_rooted_twice() {
        let dir = tempfile::tempdir().unwrap();
        let root_dir = dir.path().join("B");
        let root = OutputRoot::prepare(&root_dir).unwrap();

        // What `discover_client_files` produces for `-o B`: a path relative
        // to the current directory, already rooted at the output directory.
        let planned = PlannedOutputPath::already_planned(&root_dir.join("session.cha")).unwrap();

        assert_eq!(
            planned.verified_under(&root).unwrap(),
            root.as_path().join("session.cha")
        );
    }

    #[test]
    fn write_result_success() {
        let dir = tempfile::tempdir().unwrap();
        let mut map = HashMap::new();
        let out_path = dir.path().join("test.cha");
        map.insert("test.cha".to_string(), out_path.clone());

        let result = FileResult {
            filename: "test.cha".into(),
            content: "@Begin\n*CHI:\thello .\n@End\n".to_string(),
            content_type: ContentType::Chat,
            error: None,
            provenance: FileProvenance::NotRead,
        };

        let ok = write_result(&result, &map, dir.path()).unwrap();
        assert!(ok);
        assert!(out_path.exists());
    }

    #[test]
    fn write_result_skips_error() {
        let dir = tempfile::tempdir().unwrap();
        let map = HashMap::new();
        let result = FileResult {
            filename: "test.cha".into(),
            content: String::new(),
            content_type: ContentType::Chat,
            error: Some("processing failed".to_string()),
            provenance: FileProvenance::NotRead,
        };

        let ok = write_result(&result, &map, dir.path()).unwrap();
        assert!(!ok);
    }

    /// POLICY: `transcribe` is handed media and returns a transcript, so the
    /// fallback renames the extension. No type states which commands rename.
    #[test]
    fn resolve_output_fallback_adds_cha_for_media() {
        let dir = tempfile::tempdir().unwrap();
        let root = std::fs::canonicalize(dir.path()).unwrap();

        assert_eq!(
            resolved("audio.mp3", &HashMap::new(), dir.path()),
            root.join("audio.cha")
        );
    }

    /// POLICY: opensmile returns CSV sidecars, which keep their extension.
    #[test]
    fn resolve_output_fallback_keeps_csv() {
        let dir = tempfile::tempdir().unwrap();
        let root = std::fs::canonicalize(dir.path()).unwrap();

        assert_eq!(
            resolved("features.csv", &HashMap::new(), dir.path()),
            root.join("features.csv")
        );
    }

    #[test]
    fn resolve_output_fallback_no_extension() {
        let dir = tempfile::tempdir().unwrap();
        let root = std::fs::canonicalize(dir.path()).unwrap();

        assert_eq!(
            resolved("filename", &HashMap::new(), dir.path()),
            root.join("filename.cha")
        );
    }

    #[test]
    fn write_result_path_traversal_blocked() {
        let dir = tempfile::tempdir().unwrap();
        let map = HashMap::new();
        let result = FileResult {
            filename: "../../../escaped.cha".into(),
            content: "bad".to_string(),
            content_type: ContentType::Chat,
            error: None,
            provenance: FileProvenance::NotRead,
        };

        let err = write_result(&result, &map, dir.path()).unwrap_err();
        assert!(
            format!("{err}").contains("path escapes output directory"),
            "expected PathTraversal, got: {err}"
        );
    }

    #[test]
    fn write_result_path_traversal_nested() {
        let dir = tempfile::tempdir().unwrap();
        let map = HashMap::new();
        let result = FileResult {
            filename: "sub/../../escaped.cha".into(),
            content: "bad".to_string(),
            content_type: ContentType::Chat,
            error: None,
            provenance: FileProvenance::NotRead,
        };

        let err = write_result(&result, &map, dir.path()).unwrap_err();
        assert!(
            format!("{err}").contains("path escapes output directory"),
            "expected PathTraversal, got: {err}"
        );
    }

    #[test]
    fn write_result_path_traversal_absolute() {
        let dir = tempfile::tempdir().unwrap();
        let map = HashMap::new();
        let result = FileResult {
            filename: "/etc/stuff".into(),
            content: "bad".to_string(),
            content_type: ContentType::Csv,
            error: None,
            provenance: FileProvenance::NotRead,
        };

        let err = write_result(&result, &map, dir.path()).unwrap_err();
        assert!(
            format!("{err}").contains("path escapes output directory"),
            "expected PathTraversal, got: {err}"
        );
    }

    /// A hostile path must not leave a directory behind on its way to being
    /// refused. The lexical check runs before any `create_dir_all`.
    #[test]
    fn write_result_traversal_creates_no_directories() {
        let dir = tempfile::tempdir().unwrap();
        let escape_target = dir.path().parent().unwrap().join("batchalign-escape-probe");
        let map = HashMap::new();
        let result = FileResult {
            filename: "../batchalign-escape-probe/escaped.cha".into(),
            content: "bad".to_string(),
            content_type: ContentType::Chat,
            error: None,
            provenance: FileProvenance::NotRead,
        };

        let err = write_result(&result, &map, dir.path()).unwrap_err();
        assert!(
            format!("{err}").contains("path escapes output directory"),
            "expected PathTraversal, got: {err}"
        );
        assert!(
            !escape_target.exists(),
            "a refused write must not have created {}",
            escape_target.display()
        );
    }

    #[test]
    fn write_result_creates_nested_parent() {
        let dir = tempfile::tempdir().unwrap();
        let out_path = dir.path().join("sub").join("deep").join("file.cha");
        let mut map = HashMap::new();
        map.insert("file.cha".to_string(), out_path.clone());

        let result = FileResult {
            filename: "file.cha".into(),
            content: "@Begin\n@End\n".to_string(),
            content_type: ContentType::Chat,
            error: None,
            provenance: FileProvenance::NotRead,
        };

        let ok = write_result(&result, &map, dir.path()).unwrap();
        assert!(ok);
        assert!(out_path.exists());
        assert_eq!(
            std::fs::read_to_string(&out_path).unwrap(),
            "@Begin\n@End\n"
        );
    }

    #[test]
    fn write_result_non_chat_content_type() {
        let dir = tempfile::tempdir().unwrap();
        let map = HashMap::new();
        let result = FileResult {
            filename: "features.csv".into(),
            content: "col1,col2\n1,2\n".to_string(),
            content_type: ContentType::Csv,
            error: None,
            provenance: FileProvenance::NotRead,
        };

        let ok = write_result(&result, &map, dir.path()).unwrap();
        assert!(ok);
        let written = dir.path().join("features.csv");
        assert!(written.exists());
        assert_eq!(
            std::fs::read_to_string(written).unwrap(),
            "col1,col2\n1,2\n"
        );
    }

    #[test]
    fn write_result_empty_content() {
        let dir = tempfile::tempdir().unwrap();
        let mut map = HashMap::new();
        let out_path = dir.path().join("empty.cha");
        map.insert("empty.cha".to_string(), out_path.clone());

        let result = FileResult {
            filename: "empty.cha".into(),
            content: String::new(),
            content_type: ContentType::Chat,
            error: None,
            provenance: FileProvenance::NotRead,
        };

        let ok = write_result(&result, &map, dir.path()).unwrap();
        assert!(ok);
        assert!(out_path.exists());
        assert_eq!(std::fs::read_to_string(&out_path).unwrap(), "");
    }
}
