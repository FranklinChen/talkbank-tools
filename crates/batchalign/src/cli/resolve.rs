//! Input resolution: turn the CLI's path arguments into one validated,
//! de-duplicated input set.
//!
//! The types form a small graph, in the order a value travels:
//!
//! 1. [`InputSource`]: which input mode the user chose. Positional paths and
//!    a `--file-list` are distinct variants, so "both at once" (which used to
//!    ignore the positional paths without a word) has no representation.
//! 2. `FileListEntry` (private): one meaningful line of a list file, parsed
//!    once, with its path already resolved against the list file's directory
//!    and its [`FileListLine`] kept for error messages.
//! 3. `ExistingInput` (private): a path proven to exist, carrying the
//!    canonical identity used for de-duplication. Built only by
//!    `ExistingInput::locate`, the single place existence is checked.
//! 4. [`ResolvedInputs`]: the ordered input set. Built only from
//!    `ExistingInput`s with first occurrence winning, so it cannot hold two
//!    spellings of the same file or directory.
//!
//! Directory inputs stay directories here. They are walked later by
//! [`discover_server_inputs`](crate::cli::discover::discover_server_inputs),
//! identically for both input modes.

use std::collections::HashSet;
use std::fmt;
use std::io;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

use crate::cli::error::CliError;

/// Where the user's input paths come from.
#[derive(Debug, Clone, Copy)]
pub enum InputSource<'a> {
    /// Positional `PATHS...` arguments, resolved against the current directory.
    Paths(&'a [PathBuf]),
    /// `--file-list FILE`: paths read from a text file, one per line.
    FileList(&'a Path),
}

/// 1-based line number inside a `--file-list` file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FileListLine(NonZeroUsize);

impl FileListLine {
    /// Line number for the zero-based index `str::lines` yields.
    fn from_index(index: usize) -> Self {
        Self(NonZeroUsize::MIN.saturating_add(index))
    }

    /// The 1-based line number.
    pub fn get(self) -> usize {
        self.0.get()
    }
}

impl fmt::Display for FileListLine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// The input set: existing paths, in first-occurrence order, each file or
/// directory at most once however many spellings named it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedInputs(Vec<PathBuf>);

impl ResolvedInputs {
    /// Keep the first spelling of each distinct canonical path.
    ///
    /// Stops at the first candidate that failed to locate, so errors keep the
    /// input order the user wrote.
    fn from_candidates(
        candidates: impl IntoIterator<Item = Result<ExistingInput, CliError>>,
    ) -> Result<Self, CliError> {
        let mut seen = HashSet::new();
        let mut paths = Vec::new();
        for candidate in candidates {
            let ExistingInput { spelled, identity } = candidate?;
            if seen.insert(identity) {
                paths.push(spelled);
            }
        }
        Ok(Self(paths))
    }

    /// The inputs as written (after list-directory resolution).
    pub fn as_paths(&self) -> &[PathBuf] {
        &self.0
    }

    /// Give up the proof and take the paths.
    pub fn into_paths(self) -> Vec<PathBuf> {
        self.0
    }
}

/// Result of resolving the CLI's path arguments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedPaths {
    /// Inputs to discover and submit.
    pub inputs: ResolvedInputs,
    /// Output directory, or `None` for in-place processing.
    pub output_dir: Option<PathBuf>,
}

/// Resolve CLI path arguments into inputs and an optional output directory.
///
/// - [`InputSource::FileList`]: read the list (blank lines and `#` comments
///   skipped); relative entries resolve against the list file's directory,
///   absolute entries stay as written. A missing entry is reported with the
///   list file and line.
/// - `--in-place` or `-o`: every positional path is an input.
/// - Legacy: exactly two positional paths where the first is a directory and
///   the second is not a file means `IN_DIR OUT_DIR`.
/// - Otherwise every positional path is an input, processed in place.
///
/// In every mode the input set is de-duplicated by canonical path, keeping
/// the first occurrence.
pub fn resolve_inputs(
    source: InputSource<'_>,
    output: Option<&Path>,
    in_place: bool,
) -> Result<ResolvedPaths, CliError> {
    let explicit_output = output.map(Path::to_path_buf);
    match source {
        InputSource::FileList(list_path) => {
            let entries = read_file_list(list_path)?;
            let inputs = ResolvedInputs::from_candidates(
                entries.into_iter().map(|entry| entry.locate(list_path)),
            )?;
            Ok(ResolvedPaths {
                inputs,
                output_dir: explicit_output,
            })
        }
        InputSource::Paths([]) => Err(CliError::NoInputPaths),
        InputSource::Paths(paths) => {
            // `--in-place` and `-o` both declare every path an input, which
            // rules out the legacy two-directory reading.
            if !in_place
                && output.is_none()
                && let [in_dir, out_dir] = paths
                && in_dir.is_dir()
                && !out_dir.is_file()
            {
                let inputs =
                    ResolvedInputs::from_candidates([ExistingInput::locate_positional(in_dir)])?;
                return Ok(ResolvedPaths {
                    inputs,
                    output_dir: Some(out_dir.clone()),
                });
            }
            let inputs = ResolvedInputs::from_candidates(
                paths.iter().map(|path| ExistingInput::locate_positional(path)),
            )?;
            Ok(ResolvedPaths {
                inputs,
                output_dir: explicit_output,
            })
        }
    }
}

/// An input path proven to exist, with the canonical path that identifies it.
struct ExistingInput {
    /// The path as the user wrote it (after list-directory resolution); this
    /// is what later stages see, so output naming is unchanged.
    spelled: PathBuf,
    /// Canonical absolute path: symlinks, `.` and `..` resolved.
    identity: PathBuf,
}

impl ExistingInput {
    /// The single existence check. Canonicalizing both proves the path exists
    /// and yields its identity; `missing` builds the caller's error for a path
    /// that does not exist, since positional paths and list entries report
    /// that differently.
    fn locate(
        spelled: PathBuf,
        missing: impl FnOnce(PathBuf) -> CliError,
    ) -> Result<Self, CliError> {
        match std::fs::canonicalize(&spelled) {
            Ok(identity) => Ok(Self { spelled, identity }),
            Err(err) if is_missing(&err) => Err(missing(spelled)),
            Err(err) => Err(CliError::Io(io::Error::new(
                err.kind(),
                format!("resolve input path {}: {err}", spelled.display()),
            ))),
        }
    }

    fn locate_positional(path: &Path) -> Result<Self, CliError> {
        Self::locate(path.to_path_buf(), CliError::InputMissing)
    }
}

/// Whether an I/O error means "no such path" rather than a failure to look.
fn is_missing(err: &io::Error) -> bool {
    matches!(
        err.kind(),
        io::ErrorKind::NotFound | io::ErrorKind::NotADirectory
    )
}

/// One meaningful line of a `--file-list` file.
#[derive(Debug, Clone, PartialEq, Eq)]
struct FileListEntry {
    line: FileListLine,
    /// Entry path joined onto the list file's directory.
    path: PathBuf,
}

impl FileListEntry {
    /// Parse one raw line; `None` for blank lines and `#` comments.
    fn parse(list_dir: &Path, line: FileListLine, raw: &str) -> Option<Self> {
        let trimmed = raw.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            return None;
        }
        // `Path::join` returns its argument unchanged when that argument is
        // absolute, so absolute entries stay exactly as written.
        Some(Self {
            line,
            path: list_dir.join(trimmed),
        })
    }

    fn locate(self, list_path: &Path) -> Result<ExistingInput, CliError> {
        let line = self.line;
        ExistingInput::locate(self.path, |path| CliError::FileListEntryMissing {
            list: list_path.to_path_buf(),
            line,
            path,
        })
    }
}

/// Read and parse a `--file-list` file into at least one entry.
fn read_file_list(list_path: &Path) -> Result<Vec<FileListEntry>, CliError> {
    let text = match std::fs::read_to_string(list_path) {
        Ok(text) => text,
        Err(err) if is_missing(&err) => {
            return Err(CliError::FileListMissing(list_path.to_path_buf()));
        }
        Err(err) => {
            return Err(CliError::Io(io::Error::new(
                err.kind(),
                format!("read file list {}: {err}", list_path.display()),
            )));
        }
    };
    // A path that just read as a file has a parent; a bare file name has the
    // empty parent, which `join` treats as the current directory, which is
    // exactly where such a list lives.
    let list_dir = list_path.parent().unwrap_or(Path::new(""));
    // Some editors prefix UTF-8 text with a byte-order mark. It is encoding,
    // not part of the first path.
    let text = text.strip_prefix('\u{feff}').unwrap_or(&text);
    let entries: Vec<FileListEntry> = text
        .lines()
        .enumerate()
        .filter_map(|(index, raw)| {
            FileListEntry::parse(list_dir, FileListLine::from_index(index), raw)
        })
        .collect();
    if entries.is_empty() {
        return Err(CliError::FileListEmpty);
    }
    Ok(entries)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::args::InputKind;
    use crate::cli::discover::discover_server_inputs;
    use std::fs;

    const CHAT: &str = "@Begin\n@End\n";

    fn write(path: &Path, content: &str) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, content).unwrap();
    }

    fn resolve_list(list: &Path) -> Result<ResolvedPaths, CliError> {
        resolve_inputs(InputSource::FileList(list), None, false)
    }

    #[test]
    fn no_paths_is_error() {
        let result = resolve_inputs(InputSource::Paths(&[]), None, false);
        assert!(matches!(result, Err(CliError::NoInputPaths)));
    }

    #[test]
    fn file_list_mode_skips_comments_and_blank_lines() {
        let dir = tempfile::tempdir().unwrap();
        let f1 = dir.path().join("a.cha");
        let f2 = dir.path().join("b.cha");
        write(&f1, CHAT);
        write(&f2, CHAT);
        let list = dir.path().join("files.txt");
        write(
            &list,
            &format!("# comment\n{}\n\n  {}  \n", f1.display(), f2.display()),
        );

        let resolved = resolve_list(&list).unwrap();
        assert_eq!(resolved.inputs.as_paths(), [f1, f2]);
        assert!(resolved.output_dir.is_none());
    }

    #[test]
    fn file_list_relative_entries_resolve_against_the_list_directory() {
        // The test process runs in the crate directory, so these entries
        // exist only when read relative to the list file.
        let dir = tempfile::tempdir().unwrap();
        let lists = dir.path().join("lists");
        write(&dir.path().join("corpus").join("a.cha"), CHAT);
        write(&lists.join("b.cha"), CHAT);
        let list = lists.join("inputs.txt");
        write(&list, "../corpus/a.cha\nb.cha\n");

        let resolved = resolve_list(&list).unwrap();
        assert_eq!(
            resolved.inputs.as_paths(),
            [lists.join("../corpus/a.cha"), lists.join("b.cha")]
        );
    }

    #[test]
    fn file_list_absolute_entries_stay_absolute() {
        let list_dir = tempfile::tempdir().unwrap();
        let elsewhere = tempfile::tempdir().unwrap();
        let target = elsewhere.path().join("a.cha");
        write(&target, CHAT);
        let list = list_dir.path().join("inputs.txt");
        write(&list, &format!("{}\n", target.display()));

        let resolved = resolve_list(&list).unwrap();
        assert_eq!(resolved.inputs.as_paths(), [target]);
    }

    #[test]
    fn file_list_directory_entry_discovers_like_a_positional_directory() {
        let dir = tempfile::tempdir().unwrap();
        let corpus = dir.path().join("corpus");
        write(&corpus.join("a.cha"), CHAT);
        write(&corpus.join("sub").join("b.cha"), CHAT);
        write(&corpus.join("notes.txt"), "not chat");
        let out = dir.path().join("out");
        let list = dir.path().join("inputs.txt");
        write(&list, "corpus\n");

        let from_list = resolve_inputs(InputSource::FileList(&list), Some(&out), false).unwrap();
        let positional = resolve_inputs(
            InputSource::Paths(std::slice::from_ref(&corpus)),
            Some(&out),
            false,
        )
        .unwrap();
        assert_eq!(from_list, positional);

        let list_files = discover_server_inputs(
            from_list.inputs.as_paths(),
            from_list.output_dir.as_deref(),
            InputKind::Chat,
        )
        .unwrap();
        let positional_files = discover_server_inputs(
            positional.inputs.as_paths(),
            positional.output_dir.as_deref(),
            InputKind::Chat,
        )
        .unwrap();
        assert_eq!(list_files, positional_files);
        assert_eq!(list_files.0.len(), 2);
    }

    #[test]
    fn file_list_duplicates_collapse_to_first_occurrence() {
        let dir = tempfile::tempdir().unwrap();
        let a = dir.path().join("a.cha");
        let b = dir.path().join("b.cha");
        write(&a, CHAT);
        write(&b, CHAT);
        let list = dir.path().join("inputs.txt");
        write(
            &list,
            &format!("b.cha\na.cha\n./b.cha\n{}\nb.cha\n", a.display()),
        );

        let resolved = resolve_list(&list).unwrap();
        assert_eq!(resolved.inputs.as_paths(), [b, a]);
    }

    #[test]
    fn file_list_missing_entry_names_the_list_and_line() {
        let dir = tempfile::tempdir().unwrap();
        write(&dir.path().join("a.cha"), CHAT);
        let list = dir.path().join("inputs.txt");
        write(&list, "# header\na.cha\nmissing.cha\n");

        let err = resolve_list(&list).unwrap_err();
        let CliError::FileListEntryMissing {
            list: reported_list,
            line,
            path,
        } = &err
        else {
            panic!("expected FileListEntryMissing, got {err:?}");
        };
        assert_eq!(reported_list, &list);
        assert_eq!(line.get(), 3);
        assert_eq!(path, &dir.path().join("missing.cha"));
        assert!(
            err.to_string()
                .starts_with(&format!("{}:3: input path does not exist: ", list.display())),
            "{err}"
        );
        assert_eq!(err.exit_code(), CliError::EXIT_USAGE);
    }

    #[test]
    fn file_list_with_only_comments_is_empty() {
        let dir = tempfile::tempdir().unwrap();
        let list = dir.path().join("inputs.txt");
        write(&list, "# nothing yet\n\n   \n");
        assert!(matches!(resolve_list(&list), Err(CliError::FileListEmpty)));
    }

    #[test]
    fn missing_file_list_is_reported_as_such() {
        let dir = tempfile::tempdir().unwrap();
        let list = dir.path().join("absent.txt");
        assert!(matches!(
            resolve_list(&list),
            Err(CliError::FileListMissing(reported)) if reported == list
        ));
    }

    #[test]
    fn file_list_ignores_a_leading_byte_order_mark() {
        let dir = tempfile::tempdir().unwrap();
        write(&dir.path().join("a.cha"), CHAT);
        let list = dir.path().join("inputs.txt");
        write(&list, "\u{feff}a.cha\n");

        let resolved = resolve_list(&list).unwrap();
        assert_eq!(resolved.inputs.as_paths(), [dir.path().join("a.cha")]);
    }

    #[test]
    fn positional_duplicates_collapse_to_first_occurrence() {
        let dir = tempfile::tempdir().unwrap();
        let a = dir.path().join("a.cha");
        let b = dir.path().join("b.cha");
        write(&a, CHAT);
        write(&b, CHAT);

        let paths = [b.clone(), a.clone(), dir.path().join(".").join("b.cha")];
        let resolved = resolve_inputs(InputSource::Paths(&paths), None, true).unwrap();
        assert_eq!(resolved.inputs.as_paths(), [b, a]);
    }

    #[test]
    fn positional_missing_path_is_input_missing() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("missing.cha");
        let result = resolve_inputs(
            InputSource::Paths(std::slice::from_ref(&missing)),
            None,
            false,
        );
        assert!(matches!(result, Err(CliError::InputMissing(path)) if path == missing));
    }

    #[test]
    fn in_place_mode() {
        let dir = tempfile::tempdir().unwrap();
        let f1 = dir.path().join("a.cha");
        write(&f1, CHAT);

        let resolved =
            resolve_inputs(InputSource::Paths(std::slice::from_ref(&f1)), None, true).unwrap();
        assert_eq!(resolved.inputs.as_paths().len(), 1);
        assert!(resolved.output_dir.is_none());
    }

    #[test]
    fn legacy_two_dir_mode() {
        let dir = tempfile::tempdir().unwrap();
        let in_dir = dir.path().join("input");
        fs::create_dir(&in_dir).unwrap();
        let out_dir = dir.path().join("nonexistent_output_dir");

        let resolved = resolve_inputs(
            InputSource::Paths(&[in_dir.clone(), out_dir.clone()]),
            None,
            false,
        )
        .unwrap();

        assert_eq!(resolved.inputs.as_paths(), [in_dir]);
        assert_eq!(resolved.output_dir, Some(out_dir));
    }

    #[test]
    fn explicit_output() {
        let dir = tempfile::tempdir().unwrap();
        let f1 = dir.path().join("a.cha");
        write(&f1, CHAT);
        let out = dir.path().join("out");

        let resolved = resolve_inputs(
            InputSource::Paths(std::slice::from_ref(&f1)),
            Some(&out),
            false,
        )
        .unwrap();
        assert_eq!(resolved.inputs.as_paths().len(), 1);
        assert_eq!(resolved.output_dir, Some(out));
    }
}
