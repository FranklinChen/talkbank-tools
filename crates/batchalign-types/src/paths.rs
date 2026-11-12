//! Provenance-tracking path newtypes for the job submission system.
//!
//! Paths in batchalign3 cross machine boundaries: a client submits paths
//! from their filesystem, the server resolves them on its own filesystem.
//! These newtypes make the provenance explicit so the compiler prevents
//! accidentally reading a client path on the server's filesystem.
//!
//! # Types
//!
//! - [`ClientPath`]: path on the submitting client's machine
//! - [`ServerPath`]: path on the server's machine (safe for I/O)
//! - [`AllowedRoots`]: the folders a paths-mode job may use, from config
//! - [`AdmittedPath`]: a client path the server may use, proven inside them
//! - [`RepoRelativePath`]: path relative to a data repo root
//! - [`MediaMappingKey`]: config key for media volume lookup

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// ClientPath, from the submitting client's filesystem
// ---------------------------------------------------------------------------

/// A path on the submitting client's filesystem.
///
/// The server MUST NOT do filesystem I/O on this directly, it's metadata
/// only. The only way to use it on the server is to admit it with the
/// server's [`AllowedRoots`]: for the server's own account, any absolute
/// path; for anyone else, only one inside a folder the server was configured
/// to let paths-mode jobs use.
///
/// Deliberately does NOT implement `AsRef<Path>` to prevent accidental
/// filesystem operations.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(transparent)]
pub struct ClientPath(String);

impl ClientPath {
    /// Create a new client path.
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    /// The raw path string as submitted by the client.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Check whether this client path contains a given path component.
    ///
    /// Used for inferring the media mapping key from a client's source
    /// directory path.
    pub fn contains_component(&self, component: &str) -> bool {
        let pattern = format!("/{component}/");
        let suffix = format!("/{component}");
        self.0.contains(&pattern) || self.0.ends_with(&suffix)
    }

    /// Append a relative path component to this client path.
    ///
    /// Returns a new [`ClientPath`] with `relative` joined under this path.
    /// This is the only sanctioned way to extend a client path; it keeps the
    /// result typed as a `ClientPath`, preventing accidental I/O on the server.
    ///
    /// If `relative` is empty, returns a clone of `self`.
    ///
    /// # When to use
    ///
    /// When `source_dir` is a top-level data root (e.g.
    /// `/Volumes/data-drive/talkbank-data`) and filenames carry a
    /// repo-key prefix (`aphasia-data/English/...`), join the parent of the
    /// filename onto `source_dir` before calling `infer_media_mapping` so the
    /// repo key becomes visible in the path.
    ///
    /// # Type safety
    ///
    /// The result is a [`ClientPath`], not a [`ServerPath`].  The compiler
    /// will reject any attempt to use it for direct filesystem I/O:
    ///
    /// ```compile_fail
    /// use std::path::Path;
    /// use batchalign_types::paths::ClientPath;
    /// let base = ClientPath::new("/data");
    /// let joined = base.join(Path::new("repo/sub"));
    /// let _: &Path = joined.as_ref(); // ERROR: ClientPath has no AsRef<Path>
    /// ```
    pub fn join(&self, relative: impl AsRef<Path>) -> Self {
        let r = relative.as_ref();
        if r.as_os_str().is_empty() {
            return self.clone();
        }
        let r_str = r.to_string_lossy();
        if self.0.is_empty() {
            Self(r_str.into_owned())
        } else {
            Self(format!("{}/{}", self.0, r_str))
        }
    }

    /// Extract the portion of the path after a given component.
    ///
    /// Returns `Some(suffix)` where `suffix` is everything after `/{component}/`
    /// in the path, or `Some("")` if the path ends with `/{component}` (i.e. the
    /// client directory IS the repo root).  Returns `None` if the component is
    /// not present at all.
    ///
    /// Examples:
    /// - `"/Users/operator/chat-data/slabank-data/French/Newcastle"` → `Some("French/Newcastle")`
    /// - `"/Users/operator/chat-data/aphasia-data"` → `Some("")`  (at repo root)
    /// - `"/Users/operator/Desktop/other"` → `None`
    pub fn suffix_after_component(&self, component: &str) -> Option<&str> {
        let interior = format!("/{component}/");
        let terminal = format!("/{component}");
        if let Some((_, suffix)) = self.0.split_once(&interior) {
            Some(suffix)
        } else if self.0.ends_with(&terminal) {
            Some("")
        } else {
            None
        }
    }
}

impl std::fmt::Display for ClientPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for ClientPath {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for ClientPath {
    fn from(s: &str) -> Self {
        Self(s.to_owned())
    }
}

// ---------------------------------------------------------------------------
// ServerPath, on the server's filesystem (safe for I/O)
// ---------------------------------------------------------------------------

/// A path on the server's filesystem.
///
/// Safe for the server to read, write, and check existence.
/// Implements `AsRef<Path>` so it can be passed directly to
/// `tokio::fs::read_to_string`, `std::fs::write`, etc.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ServerPath(PathBuf);

impl ServerPath {
    /// Create from an absolute path on the server.
    pub fn new(p: impl Into<PathBuf>) -> Self {
        Self(p.into())
    }

    /// The underlying `Path` reference, safe for filesystem I/O.
    pub fn as_path(&self) -> &Path {
        &self.0
    }

    /// Join a relative component to produce a new server path.
    pub fn join(&self, component: impl AsRef<Path>) -> Self {
        Self(self.0.join(component))
    }

    /// The raw string representation.
    pub fn as_str(&self) -> &str {
        self.0.to_str().unwrap_or("")
    }
}

impl AsRef<Path> for ServerPath {
    fn as_ref(&self) -> &Path {
        &self.0
    }
}

impl std::fmt::Display for ServerPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0.display())
    }
}

impl From<PathBuf> for ServerPath {
    fn from(p: PathBuf) -> Self {
        Self(p)
    }
}

impl Default for ServerPath {
    fn default() -> Self {
        Self(PathBuf::new())
    }
}

impl From<&str> for ServerPath {
    fn from(s: &str) -> Self {
        Self(PathBuf::from(s))
    }
}

impl From<String> for ServerPath {
    fn from(s: String) -> Self {
        Self(PathBuf::from(s))
    }
}

// ---------------------------------------------------------------------------
// AllowedRoots and AdmittedPath: paths mode, admitted by configuration
// ---------------------------------------------------------------------------

/// Who may name paths-mode files, and where.
///
/// A server admits only inside the folders of its configuration
/// (`paths_mode_roots`), each absolute and resolved once, when admitted; none
/// means paths mode is off. The server has no authentication and a
/// paths-mode job names files on the server's own disk, so without this any
/// client that reached it could have it read or overwrite any file its
/// account can.
///
/// The in-process CLI runner is the user's own process acting on paths that
/// user gave it: [`AllowedRoots::same_user`] states that, rather than
/// pretending some folder (`/`, which is not even absolute on Windows) was
/// configured. Deliberately no `Default`: each of the three is chosen.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AllowedRoots(Authority);

#[derive(Debug, Clone, PartialEq, Eq)]
enum Authority {
    /// The configured folders; empty is paths mode off.
    Folders(Vec<PathBuf>),
    /// The user's own process: any absolute path without `.` or `..`.
    SameUser,
}

/// Why a configured root was refused.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RootRefusal {
    /// The root is not an absolute path.
    #[error("paths_mode_roots entry {0:?} is not an absolute path")]
    Relative(PathBuf),
    /// The root could not be resolved on this machine.
    #[error("paths_mode_roots entry {path:?} cannot be resolved: {error}")]
    Unresolvable {
        /// The configured root.
        path: PathBuf,
        /// What resolving it reported.
        error: String,
    },
}

/// Why a client path was not admitted for a paths-mode job.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PathRefusal {
    /// The server's configuration lists no paths-mode folders.
    #[error(
        "paths mode is not enabled on this server: list the folders paths-mode jobs may use under \
         paths_mode_roots in its server.yaml"
    )]
    Disabled,
    /// The path is not absolute.
    #[error("path {0} is not absolute")]
    Relative(String),
    /// The path has a `.` or `..` component.
    #[error("path {0} has a '.' or '..' component")]
    DotComponent(String),
    /// The path resolves outside every allowed folder.
    #[error(
        "path {path} is outside the folders this server allows paths-mode jobs to use ({roots})"
    )]
    Outside {
        /// The submitted path.
        path: String,
        /// The allowed folders, listed for the remedy.
        roots: String,
    },
    /// No ancestor of the path could be resolved.
    #[error("path {path} cannot be resolved: {error}")]
    Unresolvable {
        /// The submitted path.
        path: String,
        /// What resolving it reported.
        error: String,
    },
}

/// A client path the server may use for a paths-mode job: absolute, with no
/// `.` or `..`, and resolving (through any symbolic links in its existing
/// part) inside one of the [`AllowedRoots`]. Built only by
/// [`AllowedRoots::admit`]; the path is kept as submitted, so the job reads
/// and writes exactly where the client asked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdmittedPath {
    client: ClientPath,
    server: ServerPath,
    /// Which file the path named when it was admitted, as the disk tells
    /// files apart.
    identity: FileIdentity,
}

/// Which file a path names, as the disk tells files apart: the device and
/// inode of its deepest existing component (so hard links, symbolic links
/// and macOS firmlinks to one file agree), and the names below it that do
/// not exist yet, folded the way that folder's volume folds names (case only
/// where the volume ignores case; Unicode normalization on macOS, whose file
/// systems ignore it). Two admitted paths with one identity are one file;
/// two with different identities are two, including `A.cha` and `a.cha` on a
/// case-sensitive disk.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FileIdentity {
    /// The deepest existing component: device and inode on Unix; on other
    /// systems its resolved path, case-folded (Windows ignores case).
    existing: ExistingFile,
    /// The names below it that do not exist yet, folded.
    pending: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum ExistingFile {
    #[cfg(unix)]
    Inode { device: u64, inode: u64 },
    #[cfg(not(unix))]
    Resolved(String),
}

impl FileIdentity {
    /// The identity of `existing` (resolved, present on disk) with the
    /// not-yet-created `pending` names below it.
    fn of(existing: &Path, pending: &[std::ffi::OsString]) -> std::io::Result<Self> {
        let metadata = std::fs::metadata(existing)?;
        #[cfg(unix)]
        let file = {
            use std::os::unix::fs::MetadataExt;
            ExistingFile::Inode {
                device: metadata.dev(),
                inode: metadata.ino(),
            }
        };
        #[cfg(not(unix))]
        let file = {
            let _ = metadata;
            ExistingFile::Resolved(existing.to_string_lossy().to_lowercase())
        };
        let folds_case = folds_case(existing);
        let pending = pending
            .iter()
            .map(|name| fold_name(&name.to_string_lossy(), folds_case))
            .collect();
        Ok(Self {
            existing: file,
            pending,
        })
    }
}

/// Whether the volume holding `folder` ignores letter case in names.
fn folds_case(folder: &Path) -> bool {
    #[cfg(target_os = "macos")]
    {
        use std::os::unix::ffi::OsStrExt;
        let Ok(name) = std::ffi::CString::new(folder.as_os_str().as_bytes()) else {
            return true;
        };
        // `pathconf` reads a property of the volume and writes nothing; on
        // failure (-1) the macOS default, case-insensitive, is assumed.
        let sensitive = unsafe { libc::pathconf(name.as_ptr(), libc::_PC_CASE_SENSITIVE) };
        sensitive != 1
    }
    #[cfg(windows)]
    {
        let _ = folder;
        true
    }
    #[cfg(not(any(target_os = "macos", windows)))]
    {
        let _ = folder;
        false
    }
}

/// A not-yet-created name as the volume will compare it.
fn fold_name(name: &str, folds_case: bool) -> String {
    #[cfg(target_os = "macos")]
    let name: String = {
        use unicode_normalization::UnicodeNormalization;
        name.nfc().collect()
    };
    #[cfg(not(target_os = "macos"))]
    let name = name.to_owned();
    match folds_case {
        true => name.to_lowercase(),
        false => name,
    }
}

impl AdmittedPath {
    /// Which file this path named when it was admitted: for telling
    /// whether two admitted paths are one file.
    pub fn identity(&self) -> &FileIdentity {
        &self.identity
    }

    /// The path on the server, for I/O.
    pub fn server(&self) -> &ServerPath {
        &self.server
    }

    /// The path as the client submitted it, for records and display.
    pub fn client(&self) -> &ClientPath {
        &self.client
    }
}

impl AllowedRoots {
    /// No folder: paths mode is off.
    pub fn none() -> Self {
        Self(Authority::Folders(Vec::new()))
    }

    /// The user's own process (the in-process CLI runner, and test fixtures
    /// standing in for it): every absolute path without `.` or `..`, kept as
    /// given, with no folder to be inside.
    pub fn same_user() -> Self {
        Self(Authority::SameUser)
    }

    /// Admit the configured roots: each must be absolute and resolvable, and
    /// is kept resolved.
    pub fn admit_roots(roots: &[PathBuf]) -> Result<Self, RootRefusal> {
        roots
            .iter()
            .map(|root| {
                if !root.is_absolute() {
                    return Err(RootRefusal::Relative(root.clone()));
                }
                std::fs::canonicalize(root).map_err(|error| RootRefusal::Unresolvable {
                    path: root.clone(),
                    error: error.to_string(),
                })
            })
            .collect::<Result<Vec<_>, _>>()
            .map(|roots| Self(Authority::Folders(roots)))
    }

    /// Admit a client path naming a file a paths-mode job reads or writes:
    /// strictly inside a root, so its folder is inside one too.
    pub fn admit(&self, path: &ClientPath) -> Result<AdmittedPath, PathRefusal> {
        self.admit_within(path, Containment::Strict)
    }

    /// Admit a client path naming a folder a paths-mode job searches: a root
    /// itself, or inside one.
    pub fn admit_folder(&self, path: &ClientPath) -> Result<AdmittedPath, PathRefusal> {
        self.admit_within(path, Containment::RootOrInside)
    }

    fn admit_within(
        &self,
        path: &ClientPath,
        containment: Containment,
    ) -> Result<AdmittedPath, PathRefusal> {
        use std::path::Component;
        let roots = match &self.0 {
            Authority::Folders(roots) if roots.is_empty() => return Err(PathRefusal::Disabled),
            Authority::Folders(roots) => Some(roots),
            Authority::SameUser => None,
        };
        let raw = Path::new(path.as_str());
        if !raw.is_absolute() {
            return Err(PathRefusal::Relative(path.as_str().to_owned()));
        }
        if raw
            .components()
            .any(|part| matches!(part, Component::CurDir | Component::ParentDir))
        {
            return Err(PathRefusal::DotComponent(path.as_str().to_owned()));
        }
        let unresolvable = |error: std::io::Error| PathRefusal::Unresolvable {
            path: path.as_str().to_owned(),
            error: error.to_string(),
        };
        let (existing, pending) = resolve_through_existing(raw).map_err(unresolvable)?;
        let identity = FileIdentity::of(&existing, &pending).map_err(unresolvable)?;
        let resolved = pending.iter().fold(existing, |acc, part| acc.join(part));
        let admitted = |identity: FileIdentity| AdmittedPath {
            client: path.clone(),
            server: ServerPath(raw.to_path_buf()),
            identity,
        };
        let Some(roots) = roots else {
            return Ok(admitted(identity));
        };
        let inside = |root: &PathBuf| match containment {
            // A root itself is a folder, never a file a job reads or writes,
            // and an output's folder must stay inside a root.
            Containment::Strict => resolved.starts_with(root) && resolved != *root,
            Containment::RootOrInside => resolved.starts_with(root),
        };
        match roots.iter().any(inside) {
            true => Ok(admitted(identity)),
            false => Err(PathRefusal::Outside {
                path: path.as_str().to_owned(),
                roots: roots
                    .iter()
                    .map(|root| root.display().to_string())
                    .collect::<Vec<_>>()
                    .join(", "),
            }),
        }
    }
}

/// How far inside a root an admitted path must be.
#[derive(Debug, Clone, Copy)]
enum Containment {
    /// Strictly inside: a file.
    Strict,
    /// The root itself, or inside: a folder.
    RootOrInside,
}

/// The path's deepest existing component, resolved (symbolic links
/// followed), and the names below it that do not exist yet, in order: an
/// output that does not exist yet is judged by where its folder really is.
///
/// A component is treated as not yet created only when nothing at all is
/// there (`symlink_metadata` reports NotFound). Anything that is there must
/// resolve: a dangling symbolic link, whose target does not exist yet, is an
/// error rather than a name to append, since writing through it would create
/// its target wherever it points. Any other error (permission, a loop, a file
/// where a folder should be) is also an error, never a reason to judge the
/// path by an ancestor instead.
fn resolve_through_existing(path: &Path) -> std::io::Result<(PathBuf, Vec<std::ffi::OsString>)> {
    let mut existing = path;
    let mut rest = Vec::new();
    loop {
        match std::fs::symlink_metadata(existing) {
            Ok(_) => {
                let resolved = std::fs::canonicalize(existing)?;
                rest.reverse();
                return Ok((resolved, rest));
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                match (existing.parent(), existing.file_name()) {
                    (Some(parent), Some(name)) => {
                        rest.push(name.to_owned());
                        existing = parent;
                    }
                    _ => return Err(error),
                }
            }
            Err(error) => return Err(error),
        }
    }
}

// ---------------------------------------------------------------------------
// RepoRelativePath: relative to a data repo root
// ---------------------------------------------------------------------------

/// A path relative to a data repo root (e.g. `"French/Newcastle/Photos/13"`):
/// plain folder names only, so it stays under whatever root it is joined to.
///
/// Built only by [`RepoRelativePath::new`] (the wire too, through
/// `try_from`), which refuses an absolute path, a drive prefix, a leading
/// `.` and any `..`, and keeps the normalized form: joined to a root, an
/// absolute path replaces the root and `..` climbs out of it. Empty is the root itself. Valid on any machine that has the repo
/// cloned, but must be combined with a server root path to produce a
/// [`ServerPath`] for I/O.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(try_from = "String", into = "String")]
pub struct RepoRelativePath(String);

/// A path that is not a relative path of plain folder names.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0:?} is not a relative path of plain folder names (no leading '/', drive, '.' or '..')")]
pub struct RepoRelativeRefusal(pub String);

impl RepoRelativePath {
    /// Admit a relative path of plain folder names.
    pub fn new(s: impl Into<String>) -> Result<Self, RepoRelativeRefusal> {
        use std::path::Component;
        let s = s.into();
        // Kept in the normalized form the components give: `a/./b` and
        // `a//b` are `a/b` (the standard library drops a `.` that is not
        // leading), so the stored text is exactly the folders it names.
        let mut names = Vec::new();
        for component in Path::new(&s).components() {
            match component {
                Component::Normal(name) => match name.to_str() {
                    Some(name) if !name.contains('\\') => names.push(name),
                    _ => return Err(RepoRelativeRefusal(s)),
                },
                Component::RootDir
                | Component::Prefix(_)
                | Component::CurDir
                | Component::ParentDir => {
                    return Err(RepoRelativeRefusal(s));
                }
            }
        }
        Ok(Self(names.join("/")))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Combine with a server root to produce an absolute server path, which
    /// stays under the root: every component is a plain name.
    pub fn resolve_on_server(&self, root: &ServerPath) -> ServerPath {
        if self.0.is_empty() {
            root.clone()
        } else {
            root.join(&self.0)
        }
    }

    /// Append a subdirectory, itself admitted.
    pub fn join(&self, sub: &RepoRelativePath) -> Self {
        if self.0.is_empty() {
            sub.clone()
        } else if sub.0.is_empty() {
            self.clone()
        } else {
            Self(format!("{}/{}", self.0, sub.0))
        }
    }
}

impl std::fmt::Display for RepoRelativePath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl TryFrom<String> for RepoRelativePath {
    type Error = RepoRelativeRefusal;
    fn try_from(s: String) -> Result<Self, Self::Error> {
        Self::new(s)
    }
}

impl From<RepoRelativePath> for String {
    fn from(path: RepoRelativePath) -> Self {
        path.0
    }
}

// ---------------------------------------------------------------------------
// MediaMappingKey: config key, not a path
// ---------------------------------------------------------------------------

/// Key into `ServerConfig.media_mappings` (e.g. `"slabank-data"`).
///
/// Not a filesystem path, it's a logical name that maps to a
/// [`ServerPath`] via the server's configuration.
#[derive(
    Debug,
    Clone,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    Default,
    Serialize,
    Deserialize,
    utoipa::ToSchema,
)]
#[serde(transparent)]
pub struct MediaMappingKey(String);

impl MediaMappingKey {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for MediaMappingKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for MediaMappingKey {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for MediaMappingKey {
    fn from(s: &str) -> Self {
        Self(s.to_owned())
    }
}

// ---------------------------------------------------------------------------
// Media mapping inference: pure function, fully typed
// ---------------------------------------------------------------------------

/// Infer the media mapping from a client's source directory path.
///
/// Checks whether any key in `mappings` appears as a path component in
/// `client_dir`.  If found, returns the key, server root, and the
/// repo-relative subdir (everything after the key in the client path).
///
/// Example:
/// - client_dir: `/Users/operator/chat-data/slabank-data/French/Newcastle/Photos`
/// - mappings: `{"slabank-data" → "/Volumes/Other/slabank"}`
/// - returns: `("slabank-data", "/Volumes/Other/slabank", "French/Newcastle/Photos")`
pub fn infer_media_mapping<'a>(
    client_dir: &ClientPath,
    mappings: impl IntoIterator<Item = (&'a MediaMappingKey, &'a ServerPath)>,
) -> Option<(MediaMappingKey, ServerPath, RepoRelativePath)> {
    for (key, root) in mappings {
        if let Some(suffix) = client_dir.suffix_after_component(key.as_str()) {
            // A client folder whose suffix is not plain names (`..`, say)
            // infers nothing: it would name a folder outside the root.
            return Some((
                key.clone(),
                root.clone(),
                RepoRelativePath::new(suffix).ok()?,
            ));
        }
    }
    None
}

// Additional impls needed for serde(default) and validation

impl ClientPath {
    /// Whether the path is empty (not set).
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl MediaMappingKey {
    /// Whether the key is empty (not set).
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn client_path_does_not_impl_as_ref_path() {
        // This is a compile-time guarantee, ClientPath deliberately
        // does NOT implement AsRef<Path>.  If this test compiles,
        // the guarantee holds.  (We verify by using ServerPath's
        // AsRef<Path> and NOT having a corresponding ClientPath call.)
        let server = ServerPath::new("/tmp/test");
        let _: &Path = server.as_ref(); // OK, ServerPath impls AsRef<Path>
        // ClientPath does NOT have this, uncomment to verify compile error:
        // let client = ClientPath::new("/tmp/test");
        // let _: &Path = client.as_ref(); // Would not compile
    }

    #[test]
    fn server_path_as_path() {
        let sp = ServerPath::new("/Volumes/Other/slabank");
        assert_eq!(sp.as_path(), Path::new("/Volumes/Other/slabank"));
    }

    /// SURVIVES: boundary. Client paths arrive over the wire and name files
    /// on the server's disk; admission is the only route to using one, so
    /// every refusal it makes is asserted here.
    #[test]
    fn a_client_path_is_admitted_only_inside_an_allowed_root()
    -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        // Not canonicalized: on hosts whose temporary folder sits behind a
        // symbolic link, this also shows a root and a path are compared
        // resolved, while the admitted path is kept as submitted.
        let root = dir.path().join("corpus");
        let outside = dir.path().join("elsewhere");
        std::fs::create_dir_all(root.join("sub"))?;
        std::fs::create_dir_all(&outside)?;
        std::fs::write(root.join("sub/a.cha"), "")?;
        #[cfg(unix)]
        std::os::unix::fs::symlink(&outside, root.join("escape"))?;
        // A link whose target does not exist yet: writing through it would
        // create the target outside.
        #[cfg(unix)]
        std::os::unix::fs::symlink(outside.join("not-yet.cha"), root.join("dangling.cha"))?;
        let roots = AllowedRoots::admit_roots(std::slice::from_ref(&root))?;
        let at = |path: &Path| ClientPath::new(path.to_string_lossy().into_owned());

        let admitted = roots.admit(&at(&root.join("sub/a.cha")))?;
        assert_eq!(admitted.server().as_path(), root.join("sub/a.cha"));
        // An output that does not exist yet, inside.
        roots.admit(&at(&root.join("sub/new/out.cha")))?;

        assert!(matches!(
            roots.admit(&ClientPath::new("sub/a.cha")),
            Err(PathRefusal::Relative(_))
        ));
        assert!(matches!(
            roots.admit(&at(&root.join("sub/../../elsewhere/x.cha"))),
            Err(PathRefusal::DotComponent(_))
        ));
        assert!(matches!(
            roots.admit(&at(&outside.join("x.cha"))),
            Err(PathRefusal::Outside { .. })
        ));
        #[cfg(unix)]
        assert!(matches!(
            roots.admit(&at(&root.join("escape/x.cha"))),
            Err(PathRefusal::Outside { .. })
        ));
        #[cfg(unix)]
        assert!(matches!(
            roots.admit(&at(&root.join("dangling.cha"))),
            Err(PathRefusal::Unresolvable { .. })
        ));
        // One file under two spellings resolves to one identity.
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(root.join("sub"), root.join("alias"))?;
            assert_eq!(
                roots.admit(&at(&root.join("alias/a.cha")))?.identity(),
                roots.admit(&at(&root.join("sub/a.cha")))?.identity()
            );
            // A hard link is one file under two names.
            std::fs::hard_link(root.join("sub/a.cha"), root.join("sub/linked.cha"))?;
            assert_eq!(
                roots.admit(&at(&root.join("sub/linked.cha")))?.identity(),
                roots.admit(&at(&root.join("sub/a.cha")))?.identity()
            );
            // Two names for files not yet created are one file only where the
            // volume folds case.
            let upper = roots.admit(&at(&root.join("sub/New.cha")))?;
            let lower = roots.admit(&at(&root.join("sub/new.cha")))?;
            assert_eq!(
                upper.identity() == lower.identity(),
                folds_case(&root.join("sub"))
            );
        }
        assert!(matches!(
            AllowedRoots::none().admit(&at(&root.join("sub/a.cha"))),
            Err(PathRefusal::Disabled)
        ));
        assert!(matches!(
            AllowedRoots::admit_roots(&[PathBuf::from("relative")]),
            Err(RootRefusal::Relative(_))
        ));
        // The user's own process: no folder, the same shape checks.
        AllowedRoots::same_user().admit(&at(&outside.join("x.cha")))?;
        assert!(matches!(
            AllowedRoots::same_user().admit(&ClientPath::new("sub/a.cha")),
            Err(PathRefusal::Relative(_))
        ));
        assert!(matches!(
            AllowedRoots::same_user().admit(&at(&root.join("sub/../a.cha"))),
            Err(PathRefusal::DotComponent(_))
        ));
        Ok(())
    }

    #[test]
    fn client_path_contains_component() {
        let cp = ClientPath::new("/Users/operator/chat-data/slabank-data/French/Newcastle");
        assert!(cp.contains_component("slabank-data"));
        assert!(!cp.contains_component("childes-data"));
        assert!(!cp.contains_component("slabank")); // partial match rejected
    }

    #[test]
    fn client_path_suffix_after_component() {
        let cp = ClientPath::new("/Users/operator/chat-data/slabank-data/French/Newcastle/Photos");
        assert_eq!(
            cp.suffix_after_component("slabank-data"),
            Some("French/Newcastle/Photos")
        );
        assert_eq!(cp.suffix_after_component("childes-data"), None);
    }

    /// SURVIVES: boundary. A repo-relative path arrives over the wire
    /// (`media_subdir`) or is inferred from a client folder, and is joined to
    /// a server root: only plain folder names are admitted, so the result
    /// stays under the root.
    #[test]
    fn a_repo_relative_path_is_plain_names_and_stays_under_its_root()
    -> Result<(), Box<dyn std::error::Error>> {
        for refused in ["/etc", "../x", "a/../../x", "./a", "\\server\\share"] {
            assert!(RepoRelativePath::new(refused).is_err(), "{refused}");
            assert!(
                serde_json::from_str::<RepoRelativePath>(&serde_json::to_string(refused)?).is_err(),
                "{refused} on the wire"
            );
        }
        let root = ServerPath::new("/Volumes/Other/slabank");
        let rel = RepoRelativePath::new("French/Newcastle/Photos")?;
        let joined = rel.join(&RepoRelativePath::new("13")?);
        assert_eq!(
            joined.resolve_on_server(&root).as_path(),
            Path::new("/Volumes/Other/slabank/French/Newcastle/Photos/13")
        );
        assert_eq!(RepoRelativePath::default().resolve_on_server(&root), root);
        assert_eq!(RepoRelativePath::new("a/./b//c/")?.as_str(), "a/b/c");
        Ok(())
    }

    #[test]
    fn media_mapping_key_serde_roundtrip() -> Result<(), Box<dyn std::error::Error>> {
        let key = MediaMappingKey::new("slabank-data");
        let json = serde_json::to_string(&key)?;
        assert_eq!(json, "\"slabank-data\"");
        let back: MediaMappingKey = serde_json::from_str(&json)?;
        assert_eq!(back, key);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // an operator's bug: infer media mapping from client path
    // -----------------------------------------------------------------------

    #[test]
    fn infer_media_mapping_from_client_path() -> Result<(), Box<dyn std::error::Error>> {
        let client_dir =
            ClientPath::new("/Users/operator/chat-data/slabank-data/French/Newcastle/Photos");
        let mut mappings = BTreeMap::new();
        mappings.insert(
            MediaMappingKey::new("slabank-data"),
            ServerPath::new("/Volumes/Other/slabank"),
        );

        let (key, root, subdir) = infer_media_mapping(&client_dir, &mappings)
            .ok_or("Should infer slabank-data from path")?;
        assert_eq!(key.as_str(), "slabank-data");
        assert_eq!(root.as_path(), Path::new("/Volumes/Other/slabank"));
        assert_eq!(subdir.as_str(), "French/Newcastle/Photos");
        Ok(())
    }

    #[test]
    fn infer_media_mapping_no_match() {
        let client_dir = ClientPath::new("/Users/operator/Desktop/random");
        let mut mappings = BTreeMap::new();
        mappings.insert(
            MediaMappingKey::new("slabank-data"),
            ServerPath::new("/Volumes/Other/slabank"),
        );

        assert!(infer_media_mapping(&client_dir, &mappings).is_none());
    }

    #[test]
    fn infer_media_mapping_multiple_keys() -> Result<(), Box<dyn std::error::Error>> {
        let client_dir =
            ClientPath::new("/Users/operator/chat-data/childes-eng-na-data/MacWhinney/01");
        let mut mappings = BTreeMap::new();
        mappings.insert(
            MediaMappingKey::new("slabank-data"),
            ServerPath::new("/Volumes/Other/slabank"),
        );
        mappings.insert(
            MediaMappingKey::new("childes-eng-na-data"),
            ServerPath::new("/Volumes/CHILDES/CHILDES"),
        );

        let (key, root, subdir) = infer_media_mapping(&client_dir, &mappings)
            .ok_or("Should infer childes-eng-na-data from path")?;
        assert_eq!(key.as_str(), "childes-eng-na-data");
        assert_eq!(root.as_path(), Path::new("/Volumes/CHILDES/CHILDES"));
        assert_eq!(subdir.as_str(), "MacWhinney/01");
        Ok(())
    }

    /// Regression: when the client's source_dir IS the repo root (ends with the
    /// repo component, no trailing slash), suffix_after_component must return
    /// Some("") so infer_media_mapping can still fire.
    ///
    /// This happens when a --file-list batch spans multiple subdirectories
    /// (e.g. Cantonese/ and Croatian/): infer_base_dir finds the common ancestor
    /// which is the repo root itself, not a subdirectory within it.
    #[test]
    fn suffix_after_component_at_repo_root() {
        let cp = ClientPath::new("/Users/operator/chat-data/aphasia-data");
        assert_eq!(
            cp.suffix_after_component("aphasia-data"),
            Some(""),
            "path ending with the component should yield empty suffix, not None"
        );
    }

    #[test]
    fn infer_media_mapping_when_source_dir_is_repo_root() -> Result<(), Box<dyn std::error::Error>>
    {
        // Regression: source_dir at the repo root (e.g.
        // /Users/operator/chat-data/aphasia-data). Previously
        // infer_media_mapping returned None here, causing
        // "Cannot find audio file" for all aphasia-data align jobs submitted
        // from a --file-list spanning multiple subdirectories.
        let client_dir = ClientPath::new("/Users/operator/chat-data/aphasia-data");
        let mut mappings = BTreeMap::new();
        mappings.insert(
            MediaMappingKey::new("aphasia-data"),
            ServerPath::new("/Users/operator/media/aphasia"),
        );

        let (key, root, subdir) = infer_media_mapping(&client_dir, &mappings)
            .ok_or("should match even when client dir is the repo root")?;
        assert_eq!(key.as_str(), "aphasia-data");
        assert_eq!(root.as_path(), Path::new("/Users/operator/media/aphasia"));
        assert_eq!(
            subdir.as_str(),
            "",
            "repo_subdir should be empty at the root"
        );

        // Simulate the rest of the resolution: file is Cantonese/Protocol/HKU/PWA/A023.cha
        let file_subdir = "Cantonese/Protocol/HKU/PWA";
        let full_subdir = subdir.join(&RepoRelativePath::new(file_subdir)?);
        assert_eq!(full_subdir.as_str(), "Cantonese/Protocol/HKU/PWA");

        let search_dir = full_subdir.resolve_on_server(&root);
        assert_eq!(
            search_dir.as_path(),
            Path::new("/Users/operator/media/aphasia/Cantonese/Protocol/HKU/PWA")
        );
        Ok(())
    }

    #[test]
    fn full_media_resolution_through_a_subfolder_of_a_mapped_repo()
    -> Result<(), Box<dyn std::error::Error>> {
        // Simulate: client_dir = .../slabank-data/French/Newcastle/Photos
        // filename = 13/p08aul13.cha → stem = p08aul13, file_subdir = 13
        // media mapping: slabank-data → /Volumes/Other/slabank
        // Expected search: /Volumes/Other/slabank/French/Newcastle/Photos/13/p08aul13.mp3

        let client_dir =
            ClientPath::new("/Users/operator/chat-data/slabank-data/French/Newcastle/Photos");

        let mut mappings = BTreeMap::new();
        mappings.insert(
            MediaMappingKey::new("slabank-data"),
            ServerPath::new("/Volumes/Other/slabank"),
        );

        let (_, root, repo_subdir) = infer_media_mapping(&client_dir, &mappings)
            .ok_or("slabank-data should be inferred from the client dir")?;

        // File-level subdir from the filename's parent
        let file_subdir = "13";
        let full_subdir = repo_subdir.join(&RepoRelativePath::new(file_subdir)?);
        assert_eq!(full_subdir.as_str(), "French/Newcastle/Photos/13");

        let search_dir = full_subdir.resolve_on_server(&root);
        assert_eq!(
            search_dir.as_path(),
            Path::new("/Volumes/Other/slabank/French/Newcastle/Photos/13")
        );
        Ok(())
    }

    // -----------------------------------------------------------------------
    // ClientPath::join
    // -----------------------------------------------------------------------

    #[test]
    fn client_path_join_appends_relative_component() {
        let base = ClientPath::new("/Volumes/data-drive/talkbank-data");
        let joined = base.join(Path::new("aphasia-data/English/Protocol/APROCSA"));
        assert_eq!(
            joined.as_str(),
            "/Volumes/data-drive/talkbank-data/aphasia-data/English/Protocol/APROCSA"
        );
    }

    #[test]
    fn client_path_join_empty_relative_returns_clone() {
        let base = ClientPath::new("/some/path");
        let joined = base.join(Path::new(""));
        assert_eq!(joined.as_str(), "/some/path");
    }

    #[test]
    fn client_path_join_result_is_client_path_not_server_path() {
        // ClientPath::join returns ClientPath, not ServerPath, the type annotation
        // below enforces that the compiler rejects any implicit widening.
        let base = ClientPath::new("/Volumes/data-drive/talkbank-data");
        let _joined: ClientPath = base.join(Path::new("aphasia-data/English"));
    }

    // -----------------------------------------------------------------------
    // Regression: content-mode --server with top-level data root
    // -----------------------------------------------------------------------

    /// Regression: when `--server` is used the client's `source_dir` is often a
    /// top-level data root like `/Volumes/data-drive/talkbank-data` that does
    /// NOT contain the repo key.  The filenames carry the key as a prefix (e.g.
    /// `aphasia-data/English/Protocol/APROCSA/2256_T4.cha`).  Passing only
    /// `source_dir` to `infer_media_mapping` returned `None` → every file failed
    /// with "Cannot find audio file".
    ///
    /// Fix: join `source_dir` with `parent(filename)` via `ClientPath::join`
    /// before inference.  The result still stays a `ClientPath` (safe
    /// no I/O), but the repo key is now visible to `infer_media_mapping`.
    #[test]
    fn infer_media_mapping_content_mode_top_level_data_root()
    -> Result<(), Box<dyn std::error::Error>> {
        let source_dir = ClientPath::new("/Volumes/data-drive/talkbank-data");
        let filename = "aphasia-data/English/Protocol/APROCSA/2256_T4.cha";

        let mut mappings = BTreeMap::new();
        mappings.insert(
            MediaMappingKey::new("aphasia-data"),
            ServerPath::new("/Users/operator/media/aphasia"),
        );

        let file_parent = Path::new(filename)
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .ok_or("filename must have a non-empty parent directory")?;
        let infer_client = source_dir.join(file_parent);
        assert_eq!(
            infer_client.as_str(),
            "/Volumes/data-drive/talkbank-data/aphasia-data/English/Protocol/APROCSA"
        );

        let (key, root, repo_subdir) = infer_media_mapping(&infer_client, &mappings)
            .ok_or("repo key must be found after join")?;
        assert_eq!(key.as_str(), "aphasia-data");
        assert_eq!(root.as_path(), Path::new("/Users/operator/media/aphasia"));
        // repo_subdir is the correct path within the media volume, no further
        // joining needed.
        assert_eq!(repo_subdir.as_str(), "English/Protocol/APROCSA");

        let search_dir = repo_subdir.resolve_on_server(&root);
        assert_eq!(
            search_dir.as_path(),
            Path::new("/Users/operator/media/aphasia/English/Protocol/APROCSA")
        );
        Ok(())
    }

    /// Variant of the content-mode regression: file at repo root (no subdir).
    /// e.g. source_dir="/Volumes/data-drive/talkbank-data",
    ///      filename="aphasia-data/A023.cha"
    #[test]
    fn infer_media_mapping_content_mode_file_directly_in_repo()
    -> Result<(), Box<dyn std::error::Error>> {
        let source_dir = ClientPath::new("/Volumes/data-drive/talkbank-data");
        let filename = "aphasia-data/A023.cha";

        let mut mappings = BTreeMap::new();
        mappings.insert(
            MediaMappingKey::new("aphasia-data"),
            ServerPath::new("/Users/operator/media/aphasia"),
        );

        let file_parent = Path::new(filename)
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .ok_or("filename must have a non-empty parent directory")?; // parent = "aphasia-data"

        let infer_client = source_dir.join(file_parent);
        // infer_client ends with "aphasia-data", suffix_after_component returns ""
        assert_eq!(
            infer_client.as_str(),
            "/Volumes/data-drive/talkbank-data/aphasia-data"
        );

        let (_, root, repo_subdir) = infer_media_mapping(&infer_client, &mappings)
            .ok_or("aphasia-data should be inferred from the joined client path")?;
        // repo_subdir is "": file is at the root of the media volume
        assert_eq!(repo_subdir.as_str(), "");
        let search_dir = repo_subdir.resolve_on_server(&root);
        assert_eq!(
            search_dir.as_path(),
            Path::new("/Users/operator/media/aphasia")
        );
        Ok(())
    }

    /// Ensure an operator's existing scenario still works with the fixed join-first
    /// approach: source_dir already embeds the full subdir, filename has a
    /// shallow dir component.
    #[test]
    fn infer_media_mapping_joins_a_nested_file_onto_a_deep_source_dir()
    -> Result<(), Box<dyn std::error::Error>> {
        let source_dir =
            ClientPath::new("/Users/operator/chat-data/slabank-data/French/Newcastle/Photos");
        let filename = "13/p08aul13.cha";

        let mut mappings = BTreeMap::new();
        mappings.insert(
            MediaMappingKey::new("slabank-data"),
            ServerPath::new("/Volumes/Other/slabank"),
        );

        let file_parent = Path::new(filename)
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .ok_or("filename must have a non-empty parent directory")?; // "13"

        let infer_client = source_dir.join(file_parent);
        // → ".../slabank-data/French/Newcastle/Photos/13"

        let (key, root, repo_subdir) = infer_media_mapping(&infer_client, &mappings)
            .ok_or("slabank-data should be inferred from the joined client path")?;
        assert_eq!(key.as_str(), "slabank-data");
        assert_eq!(root.as_path(), Path::new("/Volumes/Other/slabank"));
        assert_eq!(repo_subdir.as_str(), "French/Newcastle/Photos/13");

        let search_dir = repo_subdir.resolve_on_server(&root);
        assert_eq!(
            search_dir.as_path(),
            Path::new("/Volumes/Other/slabank/French/Newcastle/Photos/13")
        );
        Ok(())
    }
}
