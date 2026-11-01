//! Read-only filesystem admission before any desktop job submission. Only this
//! producer can construct a path receipt; a UI proposal is not filesystem proof.

use std::collections::HashSet;
use std::fs::{self, Metadata};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use unicode_normalization::UnicodeNormalization;

#[derive(Debug, thiserror::Error)]
enum PathPlanError {
    #[error("Cannot inspect {path}: {source}")]
    Inspect {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("Choose an absolute existing folder: {0}")]
    Directory(PathBuf),
    #[error("Choose a separate output folder outside the input tree; neither folder may contain the other")]
    OverlappingTrees,
    #[error("An input is not a regular file inside the selected folder: {0}")]
    Input(PathBuf),
    #[error("The same input file was discovered through more than one name: {0}")]
    DuplicateInput(PathBuf),
    #[error("An output would overwrite a selected input: {0}")]
    InputOverwrite(PathBuf),
    #[error("An output resolves outside its selected folder: {0}")]
    OutputEscape(PathBuf),
    #[error("Several inputs would write the same output name, including case or Unicode spelling aliases: {0}")]
    OutputCollision(PathBuf),
    #[error("An output location is a directory or special file: {0}")]
    OutputKind(PathBuf),
    #[error("The desktop cannot represent this filename; rename it before processing: {0:?}")]
    Filename(PathBuf),
    #[error("Choose a folder containing compatible input files")]
    Empty,
    #[error("Save accuracy results to a separate folder so reference transcripts are preserved")]
    BenchmarkInPlace,
}

fn inspect(path: &Path, error: std::io::Error) -> PathPlanError {
    PathPlanError::Inspect {
        path: path.to_owned(),
        source: error,
    }
}

#[cfg(unix)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct FileIdentity {
    device: u64,
    inode: u64,
}

#[cfg(not(unix))]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct FileIdentity(std::sync::Arc<same_file::Handle>);

impl FileIdentity {
    fn observe(path: &Path, metadata: &Metadata) -> Result<Self, PathPlanError> {
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            let _ = path;
            Ok(Self {
                device: metadata.dev(),
                inode: metadata.ino(),
            })
        }
        #[cfg(not(unix))]
        {
            let _ = metadata;
            same_file::Handle::from_path(path)
                .map(|handle| Self(std::sync::Arc::new(handle)))
                .map_err(|error| inspect(path, error))
        }
    }
}

struct Directory {
    path: PathBuf,
    identity: FileIdentity,
}

impl Directory {
    fn admit(path: &Path) -> Result<Self, PathPlanError> {
        if !path.is_absolute() {
            return Err(PathPlanError::Directory(path.to_owned()));
        }
        let canonical = path.canonicalize().map_err(|error| inspect(path, error))?;
        let metadata = canonical
            .metadata()
            .map_err(|error| inspect(&canonical, error))?;
        if !metadata.is_dir() {
            return Err(PathPlanError::Directory(canonical));
        }
        Ok(Self {
            identity: FileIdentity::observe(&canonical, &metadata)?,
            path: canonical,
        })
    }
}

/// Desktop command subset, paired with the frontend command vocabulary.
#[derive(Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PathCommand {
    Transcribe,
    Morphotag,
    Align,
    Translate,
    Utseg,
    Benchmark,
}

impl PathCommand {
    fn output_path(self, source: &Path) -> PathBuf {
        match self {
            Self::Transcribe | Self::Benchmark => source.with_extension("cha"),
            Self::Morphotag | Self::Align | Self::Translate | Self::Utseg => source.to_owned(),
        }
    }
}

#[derive(Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
pub enum PathDestination {
    Separate { folder: PathBuf },
    InPlace,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PathProposal {
    command: PathCommand,
    source_root: PathBuf,
    inputs: Vec<PathBuf>,
    destination: PathDestination,
}

struct Source {
    path: PathBuf,
    identity: FileIdentity,
}

/// Existence is a closed observation, never an I/O-error-as-absence fallback.
enum Target {
    Existing {
        path: PathBuf,
        identity: FileIdentity,
    },
    Absent {
        path: PathBuf,
    },
}

impl Target {
    fn observe(path: &Path) -> Result<Self, PathPlanError> {
        let mut ancestor = path.to_owned();
        let mut tail = Vec::new();
        loop {
            match fs::symlink_metadata(&ancestor) {
                Ok(_) => {
                    let mut canonical = ancestor
                        .canonicalize()
                        .map_err(|error| inspect(&ancestor, error))?;
                    let metadata = canonical
                        .metadata()
                        .map_err(|error| inspect(&canonical, error))?;
                    if tail.is_empty() {
                        if !metadata.is_file() {
                            return Err(PathPlanError::OutputKind(canonical));
                        }
                        return Ok(Self::Existing {
                            identity: FileIdentity::observe(&canonical, &metadata)?,
                            path: canonical,
                        });
                    }
                    if !metadata.is_dir() {
                        return Err(PathPlanError::OutputKind(canonical));
                    }
                    for component in tail.into_iter().rev() {
                        canonical.push(component);
                    }
                    return Ok(Self::Absent { path: canonical });
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                    let name = ancestor
                        .file_name()
                        .ok_or_else(|| PathPlanError::OutputKind(ancestor.clone()))?
                        .to_owned();
                    tail.push(name);
                    if !ancestor.pop() {
                        return Err(PathPlanError::OutputKind(path.to_owned()));
                    }
                }
                Err(error) => return Err(inspect(&ancestor, error)),
            }
        }
    }

    fn path(&self) -> &Path {
        match self {
            Self::Existing { path, .. } | Self::Absent { path } => path,
        }
    }
}

fn wire(path: &Path) -> Result<String, PathPlanError> {
    path.to_str()
        .map(str::to_owned)
        .ok_or_else(|| PathPlanError::Filename(path.to_owned()))
}

/// Portable destination spelling is deliberately stricter than a case-sensitive
/// filesystem: distinct outputs may not depend on case or Unicode normalization.
#[derive(PartialEq, Eq, Hash)]
struct PortableDestination(String);

impl PortableDestination {
    fn for_path(path: &Path) -> Result<Self, PathPlanError> {
        Ok(Self(
            wire(path)?.nfd().flat_map(char::to_lowercase).collect(),
        ))
    }
}

/// Only admission can construct this serializable observation. It is not a
/// reservation against another process changing the filesystem after admission.
#[derive(Debug, Serialize)]
pub struct AdmittedPathPlan {
    source_root: String,
    inputs: Vec<String>,
    output_root: String,
    outputs: Vec<String>,
}

impl PathProposal {
    fn admit(self) -> Result<AdmittedPathPlan, PathPlanError> {
        if self.inputs.is_empty() {
            return Err(PathPlanError::Empty);
        }
        let root = Directory::admit(&self.source_root)?;
        let output = match &self.destination {
            PathDestination::Separate { folder } => {
                let output = Directory::admit(folder)?;
                if root.identity == output.identity
                    || root.path.starts_with(&output.path)
                    || output.path.starts_with(&root.path)
                {
                    return Err(PathPlanError::OverlappingTrees);
                }
                output
            }
            PathDestination::InPlace => {
                if matches!(self.command, PathCommand::Benchmark) {
                    return Err(PathPlanError::BenchmarkInPlace);
                }
                Directory::admit(&root.path)?
            }
        };
        let mut identities = HashSet::new();
        let mut sources = Vec::new();
        for requested in self.inputs {
            if !requested.is_absolute() {
                return Err(PathPlanError::Input(requested));
            }
            let path = requested
                .canonicalize()
                .map_err(|error| inspect(&requested, error))?;
            let metadata = path.metadata().map_err(|error| inspect(&path, error))?;
            if !metadata.is_file() || !path.starts_with(&root.path) {
                return Err(PathPlanError::Input(path));
            }
            let identity = FileIdentity::observe(&path, &metadata)?;
            if !identities.insert(identity.clone()) {
                return Err(PathPlanError::DuplicateInput(path));
            }
            sources.push(Source { path, identity });
        }
        let mut names = HashSet::new();
        let mut existing_outputs = HashSet::new();
        let mut outputs = Vec::new();
        for source in &sources {
            let relative = source
                .path
                .strip_prefix(&root.path)
                .map_err(|_| PathPlanError::Input(source.path.clone()))?;
            let requested = output.path.join(self.command.output_path(relative));
            let target = Target::observe(&requested)?;
            if !target.path().starts_with(&output.path) {
                return Err(PathPlanError::OutputEscape(requested));
            }
            if let Target::Existing { identity, .. } = &target {
                let same_input_in_place = matches!(&self.destination, PathDestination::InPlace)
                    && !matches!(
                        self.command,
                        PathCommand::Transcribe | PathCommand::Benchmark
                    )
                    && *identity == source.identity;
                if identities.contains(identity) && !same_input_in_place {
                    return Err(PathPlanError::InputOverwrite(requested));
                }
                if !existing_outputs.insert(identity.clone()) {
                    return Err(PathPlanError::OutputCollision(requested));
                }
            }
            if !names.insert(PortableDestination::for_path(target.path())?) {
                return Err(PathPlanError::OutputCollision(requested));
            }
            outputs.push(wire(target.path())?);
        }
        Ok(AdmittedPathPlan {
            source_root: wire(&root.path)?,
            inputs: sources
                .iter()
                .map(|source| wire(&source.path))
                .collect::<Result<_, _>>()?,
            output_root: wire(&output.path)?,
            outputs,
        })
    }
}

/// Resolve actual filesystem aliases before a job can be submitted.
#[tauri::command]
pub async fn admit_job_paths(proposal: PathProposal) -> Result<AdmittedPathPlan, String> {
    tauri::async_runtime::spawn_blocking(move || proposal.admit())
        .await
        .map_err(|error| error.to_string())?
        .map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests;
