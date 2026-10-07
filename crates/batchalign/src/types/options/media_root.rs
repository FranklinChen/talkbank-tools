//! Absolute, Unicode-preserving roots admitted before crossing daemon cwd.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// A local media-search root independent of the daemon's working directory.
/// Admission checks path representation, not existence or media suitability.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub struct AbsoluteMediaRoot(String);

/// Original wire/storage declaration, not an admitted execution root.
/// Historical relative declarations remain inspectable, never executable.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct MediaRootDeclaration(String);

impl MediaRootDeclaration {
    /// Consume the declaration's meaning at a fresh submission or plan boundary.
    /// No caller cwd is inferred from stored or API metadata.
    pub fn admit_absolute(&self) -> Result<AbsoluteMediaRoot, MediaRootRefusal> {
        AbsoluteMediaRoot::admit(self.0.as_str())
    }
}

impl From<AbsoluteMediaRoot> for MediaRootDeclaration {
    fn from(root: AbsoluteMediaRoot) -> Self {
        Self(root.0)
    }
}

/// Why a supplied root cannot become a cwd-independent wire value.
#[derive(Debug, thiserror::Error)]
pub enum MediaRootRefusal {
    /// An empty path is not an explicit media root.
    #[error("media directory must not be empty")]
    Empty,
    /// Wire roots and CLI resolution bases must already be absolute.
    #[error("media directory must be absolute before submission: {path}", path = .0.display())]
    Relative(
        /// The unresolved path, preserved for diagnostics.
        PathBuf,
    ),
    /// JSON cannot represent this path without changing its bytes.
    #[error("media directory is not Unicode and cannot be represented losslessly in JSON: {path}", path = .0.display())]
    NonUnicode(
        /// Original bytes, never replaced in an admitted path.
        PathBuf,
    ),
}

impl AbsoluteMediaRoot {
    /// Admit an already-absolute path without canonicalizing, normalizing or
    /// requiring that it exists. Relative wire values must not gain a cwd.
    pub fn admit(path: impl Into<PathBuf>) -> Result<Self, MediaRootRefusal> {
        let path = path.into();
        if path.as_os_str().is_empty() {
            return Err(MediaRootRefusal::Empty);
        }
        if !path.is_absolute() {
            return Err(MediaRootRefusal::Relative(path));
        }
        let value = path
            .to_str()
            .ok_or_else(|| MediaRootRefusal::NonUnicode(path.clone()))?;
        Ok(Self(value.to_owned()))
    }

    /// Anchor a CLI-relative root to an explicit submission cwd, preserving
    /// symlinks and `..` rather than guessing a filesystem canonical identity.
    pub fn resolve_cli(path: &Path, cwd: &Path) -> Result<Self, MediaRootRefusal> {
        if path.as_os_str().is_empty() {
            return Err(MediaRootRefusal::Empty);
        }
        if path.is_absolute() {
            return Self::admit(path);
        }
        let base = Self::admit(cwd)?;
        Self::admit(base.as_path().join(path))
    }

    /// Borrow the admitted root for local media-search I/O.
    pub fn as_path(&self) -> &Path {
        Path::new(&self.0)
    }
}

impl<'de> Deserialize<'de> for AbsoluteMediaRoot {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let path = String::deserialize(deserializer)?;
        Self::admit(path).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn media_root_cli_resolution_is_source_bound_without_fs_or_cwd_mutation() {
        let temp = tempfile::tempdir().expect("base");
        let root = AbsoluteMediaRoot::resolve_cli(Path::new("media/../声 recordings"), temp.path())
            .expect("anchored root");
        assert_eq!(root.as_path(), temp.path().join("media/../声 recordings"));
        assert!(!root.as_path().exists());
        assert!(matches!(
            AbsoluteMediaRoot::resolve_cli(Path::new("media"), Path::new("relative")),
            Err(MediaRootRefusal::Relative(_))
        ));
        assert!(matches!(
            AbsoluteMediaRoot::resolve_cli(Path::new(""), temp.path()),
            Err(MediaRootRefusal::Empty)
        ));
    }

    #[test]
    fn media_root_wire_stays_a_string_but_refuses_relative_and_empty_roots() {
        let temp = tempfile::tempdir().expect("base");
        let expected = temp.path().join("é 声 recordings");
        let admitted = AbsoluteMediaRoot::admit(expected.clone()).expect("absolute root");
        let wire = serde_json::to_value(&admitted).expect("wire");
        assert_eq!(wire.as_str(), expected.to_str());
        assert_eq!(
            serde_json::from_value::<AbsoluteMediaRoot>(wire).expect("readback"),
            admitted
        );
        for value in ["", ".", "media", "../media"] {
            assert!(serde_json::from_value::<AbsoluteMediaRoot>(serde_json::json!(value)).is_err());
        }
    }

    #[test]
    fn media_root_historical_declarations_remain_readable_not_admitted() {
        for value in ["", ".", "media", "../media"] {
            let declaration: MediaRootDeclaration =
                serde_json::from_value(serde_json::json!(value)).expect("historical metadata");
            assert_eq!(
                serde_json::to_value(&declaration).unwrap(),
                serde_json::json!(value)
            );
            assert!(declaration.admit_absolute().is_err());
        }
    }

    #[cfg(unix)]
    #[test]
    fn media_root_refuses_non_unicode_instead_of_lossy_replacement() {
        use std::os::unix::ffi::OsStringExt;
        let temp = tempfile::tempdir().expect("base");
        let invalid = temp.path().join(std::ffi::OsString::from_vec(vec![0xff]));
        assert!(matches!(
            AbsoluteMediaRoot::admit(invalid),
            Err(MediaRootRefusal::NonUnicode(_))
        ));
    }
}
