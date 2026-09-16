//! File discovery: mirrors `file_io.py` (`_discover_files`, `_discover_inputs`).
//!
//! Walks directories, filters by extension, sorts by size (largest first),
//! detects and skips dummy CHAT files.

use std::collections::HashSet;
use std::ffi::OsStr;
use std::fs;
use std::io;
use std::io::Read;
use std::path::{Path, PathBuf};

use walkdir::WalkDir;

use crate::ReleasedCommand;

use crate::cli::args::InputKind;
use crate::cli::error::CliError;

/// Check whether a CHAT file is a "dummy" placeholder that should be copied,
/// not processed.
///
/// Reads the first 512 bytes and checks for the `@Options:\tdummy` header or
/// the standard TalkBank dummy-file text.
pub fn is_dummy_chat(path: &Path) -> bool {
    let Ok(mut f) = fs::File::open(path) else {
        return false;
    };
    let mut buf = [0u8; 512];
    let n = match f.read(&mut buf) {
        Ok(n) => n,
        Err(_) => return false,
    };
    let text = String::from_utf8_lossy(&buf[..n]);
    text.contains("@Options:\tdummy")
        || text.contains("This is a dummy file to permit playback from the TalkBank browser")
}

fn is_appledouble_sidecar(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.starts_with("._"))
}

/// Discover files from a single directory for server dispatch.
///
/// Walks `in_dir` recursively, filters by `input_kind`, sorts by file size
/// (largest first). Dummy CHAT files are skipped (should be copied separately).
///
/// Returns `(files, outputs)` where `outputs[i]` is the output path for `files[i]`.
pub fn discover_client_files(
    in_dir: &Path,
    out_dir: &Path,
    input_kind: InputKind,
) -> Result<(Vec<PathBuf>, Vec<PathBuf>), CliError> {
    let mut files = Vec::new();
    let mut outputs = Vec::new();

    for entry in WalkDir::new(in_dir) {
        let entry = entry.map_err(walkdir_error)?;
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path();
        if is_appledouble_sidecar(path) {
            continue;
        }
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        // Skip dummy CHAT files
        if ext == "cha" && is_dummy_chat(path) {
            // Copy to output (unless in-place)
            if in_dir != out_dir {
                let rel = path.strip_prefix(in_dir).map_err(|err| {
                    invalid_data(format!(
                        "failed to derive path relative to {} for {}: {err}",
                        in_dir.display(),
                        path.display()
                    ))
                })?;
                let dest = out_dir.join(rel);
                if let Some(parent) = dest.parent() {
                    fs::create_dir_all(parent).map_err(|err| {
                        io_with_path("create dummy output directory", parent, err)
                    })?;
                }
                fs::copy(path, &dest)
                    .map_err(|err| io_with_path("copy dummy CHAT file", &dest, err))?;
            }
            continue;
        }

        if input_kind.accepts(&ext) {
            let rel = path.strip_prefix(in_dir).map_err(|err| {
                invalid_data(format!(
                    "failed to derive path relative to {} for {}: {err}",
                    in_dir.display(),
                    path.display()
                ))
            })?;
            let out_path = out_dir.join(rel);
            files.push(path.to_path_buf());
            outputs.push(out_path);
        }
    }

    // Sort by file size (largest first) to avoid stragglers
    sort_by_size_desc(&mut files, &mut outputs)?;

    Ok((files, outputs))
}

/// Files discovered across several inputs, each kept once.
///
/// A directory input plus a file inside it (or two nested directories) reach
/// the same file twice; submitting it twice would process and write it twice.
/// The canonical path is the identity, and the first input to reach a file
/// decides its output path.
#[derive(Default)]
struct UniqueDiscoveries {
    seen: HashSet<PathBuf>,
    files: Vec<PathBuf>,
    outputs: Vec<PathBuf>,
}

impl UniqueDiscoveries {
    fn push(&mut self, file: PathBuf, output: PathBuf) -> Result<(), CliError> {
        let identity = canonicalize_path(&file, "canonicalize discovered input file")?;
        if self.seen.insert(identity) {
            self.files.push(file);
            self.outputs.push(output);
        }
        Ok(())
    }
}

/// Discover files from mixed inputs (directories + individual files) for server dispatch.
///
/// For directories: walks recursively via [`discover_client_files`].
/// For individual files: adds directly (no extension filtering, user chose them).
/// A file reached by more than one input is kept once, at its first occurrence.
pub fn discover_server_inputs(
    inputs: &[PathBuf],
    out_dir: Option<&Path>,
    input_kind: InputKind,
) -> Result<(Vec<PathBuf>, Vec<PathBuf>), CliError> {
    let mut discovered = UniqueDiscoveries::default();

    for inp in inputs {
        let inp_path = Path::new(inp);
        if inp_path.is_dir() {
            let d_out = out_dir
                .map(PathBuf::from)
                .unwrap_or_else(|| inp_path.to_path_buf());
            let (fs, os) = discover_client_files(inp_path, &d_out, input_kind)?;
            for (file, output) in fs.into_iter().zip(os) {
                discovered.push(file, output)?;
            }
        } else if inp_path.is_file() {
            let out_path = if let Some(od) = out_dir {
                let name = required_file_name(inp_path)?;
                PathBuf::from(od).join(name)
            } else {
                inp_path.to_path_buf() // in-place
            };
            discovered.push(inp_path.to_path_buf(), out_path)?;
        } else {
            return Err(CliError::InputMissing(inp_path.to_path_buf()));
        }
    }

    let UniqueDiscoveries {
        files: mut all_files,
        outputs: mut all_outputs,
        ..
    } = discovered;

    // Sort by file size (largest first)
    sort_by_size_desc(&mut all_files, &mut all_outputs)?;

    Ok((all_files, all_outputs))
}

/// Sort two parallel vectors by file size (largest first).
fn sort_by_size_desc(files: &mut Vec<PathBuf>, outputs: &mut Vec<PathBuf>) -> Result<(), CliError> {
    if files.is_empty() {
        return Ok(());
    }
    let mut pairs = Vec::new();
    for (file, output) in files.drain(..).zip(outputs.drain(..)) {
        let size = fs::metadata(&file)
            .map_err(|err| io_with_path("read file metadata during discovery sort", &file, err))?
            .len();
        pairs.push((file, output, size));
    }
    pairs.sort_by_key(|b| std::cmp::Reverse(b.2));
    for (f, o, _) in pairs {
        files.push(f);
        outputs.push(o);
    }
    Ok(())
}

/// Infer a base directory from the inputs list for media mapping detection.
///
/// For directory inputs: returns the first directory.
/// For individual files: returns the common ancestor directory.
pub fn infer_base_dir(inputs: &[PathBuf]) -> Result<PathBuf, CliError> {
    let dirs: Vec<&Path> = inputs
        .iter()
        .map(|s| s.as_path())
        .filter(|p| p.is_dir())
        .collect();

    if let Some(&d) = dirs.first() {
        return canonicalize_path(d, "canonicalize input directory");
    }

    // All inputs are files, common ancestor
    if !inputs.is_empty() {
        let abs: Vec<PathBuf> = inputs
            .iter()
            .map(|p| canonicalize_path(Path::new(p), "canonicalize input file"))
            .collect::<Result<_, _>>()?;
        if abs.len() > 1 {
            // Find common prefix
            if let Some(first) = abs.first() {
                let mut common = first.clone();
                for path in &abs[1..] {
                    while !path.starts_with(&common) {
                        if !common.pop() {
                            break;
                        }
                    }
                }
                if common.is_file()
                    && let Some(parent) = common.parent()
                {
                    return Ok(parent.to_path_buf());
                }
                return Ok(common);
            }
        } else if let Some(first) = abs.first()
            && let Some(parent) = first.parent()
        {
            return Ok(parent.to_path_buf());
        }
    }

    Ok(PathBuf::from("."))
}

/// Build unique relative names for server payload and a result mapping.
///
/// Returns `(server_names, result_map)` where `result_map[server_name] = output_path`.
pub fn build_server_names(
    files: &[PathBuf],
    outputs: &[PathBuf],
    inputs: &[PathBuf],
) -> Result<(Vec<String>, std::collections::HashMap<String, PathBuf>), CliError> {
    use std::collections::HashMap;

    let dir_inputs: Vec<PathBuf> = inputs
        .iter()
        .filter(|p| p.is_dir())
        .map(|p| {
            canonicalize_path(
                Path::new(p),
                "canonicalize input directory for server naming",
            )
        })
        .collect::<Result<_, _>>()?;

    // Find individual files (not under any dir input)
    let individual_abs: Vec<PathBuf> = files
        .iter()
        .map(|f| canonicalize_path(f, "canonicalize input file for server naming"))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .filter(|abs| !dir_inputs.iter().any(|d| abs.strip_prefix(d).is_ok()))
        .collect();

    // Common ancestor for individual files
    let common: PathBuf = if individual_abs.len() > 1 {
        let mut c = individual_abs[0].clone();
        for path in &individual_abs[1..] {
            while !path.starts_with(&c) {
                if !c.pop() {
                    break;
                }
            }
        }
        if c.is_file() {
            c.parent().map(|p| p.to_path_buf()).unwrap_or(c)
        } else {
            c
        }
    } else if let Some(first) = individual_abs.first() {
        first.parent().map(|p| p.to_path_buf()).unwrap_or_default()
    } else {
        PathBuf::from("/")
    };

    let mut server_names = Vec::with_capacity(files.len());
    let mut result_map = HashMap::with_capacity(files.len());

    for (fpath, opath) in files.iter().zip(outputs.iter()) {
        let abs = canonicalize_path(fpath, "canonicalize input file for dispatch")?;

        // Check if this file is under a directory input
        let mut rel: Option<String> = None;
        for d in &dir_inputs {
            if let Ok(suffix) = abs.strip_prefix(d) {
                rel = Some(suffix.to_string_lossy().to_string());
                break;
            }
        }

        let rel = match rel {
            Some(rel) => rel,
            None => match abs.strip_prefix(&common) {
                Ok(path) => path.to_string_lossy().to_string(),
                Err(_) => required_file_name(&abs)?.to_string_lossy().to_string(),
            },
        };

        // Normalize Windows backslashes to forward slashes so display names
        // are platform-independent (server always runs on Unix).
        let rel = if rel.contains('\\') {
            rel.replace('\\', "/")
        } else {
            rel
        };
        server_names.push(rel.clone());
        result_map.insert(rel, opath.clone());
    }

    Ok((server_names, result_map))
}

/// Commands that create new files from media input.
///
/// These should never copy non-matching files to output.
pub const GENERATION_COMMANDS: &[ReleasedCommand] = &[
    ReleasedCommand::Transcribe,
    ReleasedCommand::TranscribeS,
    ReleasedCommand::Benchmark,
    ReleasedCommand::Opensmile,
];

/// Why a non-matching file traveled through to the output directory
/// unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CopyReason {
    /// Not one of this command's input extensions, and not a record about
    /// the input directory itself (media alongside a `.cha` input, a
    /// README, or any other file the command does not consume).
    NotCommandInput,
}

/// Why a non-matching file was withheld from the output directory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkipReason {
    /// Named as a record describing the INPUT directory as a whole (a
    /// provenance manifest written by an upstream pipeline stage, for
    /// example). Copying it unchanged into the output directory would leave
    /// a record there that describes a different artifact than the one this
    /// run actually produced: exactly the misattribution a provenance
    /// record exists to prevent.
    InputDirectoryRecord,
}

/// The passthrough decision for one non-matching file found while walking
/// the input directory.
///
/// A sum type rather than a bool: "does this file travel through" and "why"
/// are one fact, and a caller reporting what was withheld needs the reason,
/// not a bit stripped of its provenance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PassthroughDecision {
    /// Copied unchanged into the output directory.
    CopiedThrough(CopyReason),
    /// Withheld: never written into the output directory.
    NotCopied(SkipReason),
}

/// One non-matching file discovered during a `copy_nonmatching` walk,
/// paired with the decision made about it, relative to the input directory
/// that was walked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PassthroughEntry {
    /// The file's path relative to the input directory that was walked.
    pub relative_path: PathBuf,
    /// What happened to it.
    pub decision: PassthroughDecision,
}

/// The full record of a `copy_nonmatching` run: every non-matching file it
/// found and what happened to each one.
///
/// Returned rather than only logged, so a caller can state in its own
/// report what was withheld and why.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PassthroughReport {
    /// Every non-matching file this walk found, in the order encountered.
    pub entries: Vec<PassthroughEntry>,
}

impl PassthroughReport {
    /// Entries this run withheld from the output directory.
    pub fn skipped(&self) -> impl Iterator<Item = &PassthroughEntry> {
        self.entries
            .iter()
            .filter(|entry| matches!(entry.decision, PassthroughDecision::NotCopied(_)))
    }

    /// Fold another run's report into this one (used when several input
    /// directories are walked for one command invocation).
    pub fn extend_from(&mut self, other: PassthroughReport) {
        self.entries.extend(other.entries);
    }
}

/// Classify one file that `input_kind` does not accept as this command's
/// input, deciding whether it travels through to the output directory.
fn classify_nonmatching(path: &Path) -> PassthroughDecision {
    if is_input_directory_record(path) {
        PassthroughDecision::NotCopied(SkipReason::InputDirectoryRecord)
    } else {
        PassthroughDecision::CopiedThrough(CopyReason::NotCommandInput)
    }
}

/// Whether `path`'s name marks it as a record describing the input
/// directory as a whole, rather than content belonging to one session.
///
/// Matched by name because these records are written by pipeline stages
/// outside this crate, not by anything under our control; a byte-content
/// sniff would have to trust the same producers this check exists to
/// distrust.
fn is_input_directory_record(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
        return false;
    };
    // Case-folded, the way filter_files_for_command matches name-based
    // rules: a record spelled Provenance.json is the same record.
    let lower = name.to_ascii_lowercase();
    lower == "provenance.json" || lower.ends_with(".provenance.json")
}

/// Copy files that are not inputs for `input_kind` from `in_dir` to `out_dir`.
///
/// Preserves relative directory structure. Skipped for in-place mode and
/// generation commands. A non-matching file that is itself a record about
/// the input directory (a provenance manifest) is never copied through: see
/// [`PassthroughDecision`].
pub fn copy_nonmatching(
    in_dir: &Path,
    out_dir: &Path,
    input_kind: InputKind,
    command: ReleasedCommand,
) -> Result<PassthroughReport, CliError> {
    if GENERATION_COMMANDS.contains(&command) {
        return Ok(PassthroughReport::default());
    }
    if let (Ok(a), Ok(b)) = (fs::canonicalize(in_dir), fs::canonicalize(out_dir))
        && a == b
    {
        return Ok(PassthroughReport::default());
    }

    let mut report = PassthroughReport::default();

    for entry in WalkDir::new(in_dir) {
        let entry = entry.map_err(walkdir_error)?;
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path();
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();
        // A record describing the input directory is withheld BEFORE the
        // input-kind gate, so a command whose input kind accepts every
        // extension can never take the record as one of its inputs.
        if !input_kind.accepts(&ext) || is_input_directory_record(path) {
            let rel = path.strip_prefix(in_dir).map_err(|err| {
                invalid_data(format!(
                    "failed to derive non-matching relative path from {} for {}: {err}",
                    in_dir.display(),
                    path.display()
                ))
            })?;
            let decision = classify_nonmatching(path);
            match decision {
                PassthroughDecision::CopiedThrough(_) => {
                    let dest = out_dir.join(rel);
                    if let Some(parent) = dest.parent() {
                        fs::create_dir_all(parent).map_err(|err| {
                            io_with_path("create output directory for copied file", parent, err)
                        })?;
                    }
                    fs::copy(path, &dest)
                        .map_err(|err| io_with_path("copy non-matching file", &dest, err))?;
                }
                PassthroughDecision::NotCopied(_) => {}
            }
            report.entries.push(PassthroughEntry {
                relative_path: rel.to_path_buf(),
                decision,
            });
        }
    }
    Ok(report)
}

/// Render a run's skip decisions for the operator, one line per withheld
/// file. A no-op when nothing was withheld.
pub fn print_passthrough_skips(report: &PassthroughReport) {
    for entry in report.skipped() {
        let PassthroughDecision::NotCopied(reason) = entry.decision else {
            continue;
        };
        let reason_text = match reason {
            SkipReason::InputDirectoryRecord => {
                "describes the input directory, not this run's output"
            }
        };
        eprintln!(
            "note: not copying {} to output ({reason_text})",
            entry.relative_path.display()
        );
    }
}

fn canonicalize_path(path: &Path, action: &'static str) -> Result<PathBuf, CliError> {
    fs::canonicalize(path).map_err(|err| io_with_path(action, path, err))
}

fn required_file_name(path: &Path) -> Result<&OsStr, CliError> {
    path.file_name().ok_or_else(|| {
        invalid_data(format!(
            "path does not have a filename component: {}",
            path.display()
        ))
    })
}

fn io_with_path(action: &'static str, path: &Path, err: io::Error) -> CliError {
    CliError::Io(io::Error::new(
        err.kind(),
        format!("{action} {}: {err}", path.display()),
    ))
}

fn walkdir_error(err: walkdir::Error) -> CliError {
    let detail = if let Some(path) = err.path() {
        format!("walk directory entry {}: {err}", path.display())
    } else {
        format!("walk directory entry: {err}")
    };
    CliError::Io(io::Error::other(detail))
}

fn invalid_data(detail: String) -> CliError {
    CliError::Io(io::Error::new(io::ErrorKind::InvalidData, detail))
}

#[cfg(test)]
mod tests;
