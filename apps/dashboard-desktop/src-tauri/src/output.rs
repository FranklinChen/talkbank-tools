//! The desktop can reveal an existing directory, never execute an arbitrary
//! path or use a generic URL/shell-opening capability.

use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug, thiserror::Error)]
enum OpenDirectoryError {
    #[error("Cannot inspect the output directory: {0}")]
    Inspect(#[source] std::io::Error),
    #[error("The output location is not an absolute directory path")]
    NotAbsolute,
    #[error("The output location is not an existing directory")]
    NotDirectory,
    #[error("Could not launch the file manager: {0}")]
    Launch(#[source] std::io::Error),
    #[error("The file manager refused the output directory ({0})")]
    Refused(std::process::ExitStatus),
}

/// Only admission can create a directory eligible for the native opener.
struct OutputDirectory(PathBuf);

impl OutputDirectory {
    fn admit(path: &Path) -> Result<Self, OpenDirectoryError> {
        if !path.is_absolute() {
            return Err(OpenDirectoryError::NotAbsolute);
        }
        let resolved = path.canonicalize().map_err(OpenDirectoryError::Inspect)?;
        if !resolved
            .metadata()
            .map_err(OpenDirectoryError::Inspect)?
            .is_dir()
        {
            return Err(OpenDirectoryError::NotDirectory);
        }
        Ok(Self(resolved))
    }

    fn reveal(self) -> Result<(), OpenDirectoryError> {
        #[cfg(target_os = "macos")]
        let mut command = Command::new("/usr/bin/open");
        #[cfg(target_os = "windows")]
        let mut command = Command::new("explorer.exe");
        #[cfg(not(any(target_os = "macos", target_os = "windows")))]
        let mut command = Command::new("xdg-open");
        // The canonical path is absolute, is passed as one argv element, and
        // cannot be an option or URL. No shell interprets its characters.
        let status = command
            .arg(self.0)
            .status()
            .map_err(OpenDirectoryError::Launch)?;
        if status.success() {
            Ok(())
        } else {
            Err(OpenDirectoryError::Refused(status))
        }
    }
}

/// Reveal a job's actual output directory; a native refusal remains visible.
#[tauri::command]
pub async fn open_output_directory(path: String) -> Result<(), String> {
    tauri::async_runtime::spawn_blocking(move || OutputDirectory::admit(Path::new(&path))?.reveal())
        .await
        .map_err(|error| error.to_string())?
        .map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_admission_rejects_relative_missing_and_file_locations() {
        assert!(matches!(
            OutputDirectory::admit(Path::new("relative")),
            Err(OpenDirectoryError::NotAbsolute)
        ));
        let temporary = std::env::temp_dir().canonicalize().unwrap();
        assert!(OutputDirectory::admit(&temporary).is_ok());
        assert!(OutputDirectory::admit(&temporary.join("missing-output-unique/child")).is_err());
        assert!(matches!(
            OutputDirectory::admit(&std::env::current_exe().unwrap()),
            Err(OpenDirectoryError::NotDirectory)
        ));
    }
}
