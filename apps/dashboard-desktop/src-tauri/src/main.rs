//! Tauri desktop shell for Batchalign3.
//!
//! Provides native file dialog integration, a `discover_files` command for
//! enumerating input files, non-owning managed-service discovery,
//! credential-preserving settings, directory-only output opening, and one
//! explicit protocol module that mirrors the frontend's desktop boundary.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod config;
mod output;
mod path_plan;
mod protocol;
mod server;

use std::path::Path;

#[derive(Debug, thiserror::Error)]
enum DiscoveryError {
    #[error("Not an absolute directory: {0}")]
    NotDirectory(std::path::PathBuf),
    #[error("Cannot inspect {path}: {source}")]
    Inspect {
        path: std::path::PathBuf,
        source: std::io::Error,
    },
    #[error("The desktop cannot represent this filename; rename it before processing: {0:?}")]
    Filename(std::path::PathBuf),
}

enum InputEntry {
    Directory(std::path::PathBuf),
    RegularFile(std::path::PathBuf),
    Ignored,
}

impl InputEntry {
    fn inspect(entry: std::fs::DirEntry) -> Result<Self, DiscoveryError> {
        let path = entry.path();
        let kind = entry
            .file_type()
            .map_err(|source| DiscoveryError::Inspect {
                path: path.clone(),
                source,
            })?;
        Ok(if kind.is_dir() {
            Self::Directory(path)
        } else if kind.is_file() {
            Self::RegularFile(path)
        } else {
            Self::Ignored
        })
    }
}

/// Enumerate files under one root directory for the desktop shell.
///
/// The helper stays separate from the Tauri command so the shell can unit-test
/// the directory contract without standing up a full desktop app instance.
fn discover_files_in_dir(
    root: &Path,
    extensions: &[String],
) -> Result<Vec<String>, DiscoveryError> {
    if !root.is_absolute() || !root.is_dir() {
        return Err(DiscoveryError::NotDirectory(root.to_owned()));
    }

    let exts: Vec<String> = extensions.iter().map(|e| e.to_lowercase()).collect();
    let mut files = Vec::new();

    fn walk(dir: &Path, exts: &[String], out: &mut Vec<String>) -> Result<(), DiscoveryError> {
        let entries = std::fs::read_dir(dir).map_err(|source| DiscoveryError::Inspect {
            path: dir.to_owned(),
            source,
        })?;

        for entry in entries {
            let entry = entry.map_err(|source| DiscoveryError::Inspect {
                path: dir.to_owned(),
                source,
            })?;
            match InputEntry::inspect(entry)? {
                InputEntry::Directory(path) => walk(&path, exts, out)?,
                InputEntry::RegularFile(path) => {
                    if exts.is_empty()
                        || path
                            .extension()
                            .and_then(|ext| ext.to_str())
                            .is_some_and(|ext| exts.contains(&ext.to_lowercase()))
                    {
                        out.push(
                            path.to_str()
                                .ok_or_else(|| DiscoveryError::Filename(path.clone()))?
                                .to_owned(),
                        );
                    }
                }
                InputEntry::Ignored => {}
            }
        }
        Ok(())
    }

    walk(root, &exts, &mut files)?;
    files.sort();
    Ok(files)
}

/// Walk a directory and return paths matching the given extensions.
///
/// The batchalign server's `POST /jobs` endpoint in paths_mode requires individual
/// file paths in `source_paths`. This command bridges the gap between the user
/// picking a folder and the API needing file paths.
#[tauri::command]
fn discover_files(dir: String, extensions: Vec<String>) -> Result<Vec<String>, String> {
    discover_files_in_dir(Path::new(&dir), &extensions).map_err(|error| error.to_string())
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            discover_files,
            path_plan::admit_job_paths,
            output::open_output_directory,
            server::start_server,
            server::server_status,
            config::is_first_launch,
            config::read_config,
            config::write_config,
        ])
        .run(tauri::generate_context!())
        .expect("error while running batchalign desktop");
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;

    use super::*;

    #[cfg(unix)]
    #[test]
    fn discovery_ignores_link_loops_outside_links_and_special_files() {
        use std::os::unix::fs::symlink;
        use std::os::unix::net::UnixListener;
        let root = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("real.cha"), "kept").unwrap();
        std::fs::write(outside.path().join("outside.cha"), "outside").unwrap();
        symlink(root.path(), root.path().join("loop")).unwrap();
        symlink(outside.path(), root.path().join("linked-folder")).unwrap();
        symlink(
            outside.path().join("outside.cha"),
            root.path().join("linked.cha"),
        )
        .unwrap();
        let _socket = UnixListener::bind(root.path().join("socket.cha")).unwrap();
        assert_eq!(
            discover_files_in_dir(root.path(), &["cha".into()]).unwrap(),
            [root.path().join("real.cha").to_str().unwrap()]
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn discovery_refuses_non_unicode_names_instead_of_fabricating_paths() {
        use std::os::unix::ffi::OsStringExt;
        let root = tempfile::tempdir().unwrap();
        let name = std::ffi::OsString::from_vec(b"invalid-\xff.cha".to_vec());
        std::fs::write(root.path().join(name), "kept").unwrap();
        assert!(matches!(
            discover_files_in_dir(root.path(), &["cha".into()]),
            Err(DiscoveryError::Filename(_))
        ));
    }

    #[test]
    fn unfiltered_discovery_includes_regular_extensionless_files() {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("README"), "kept").unwrap();
        assert_eq!(discover_files_in_dir(root.path(), &[]).unwrap().len(), 1);
    }

    struct TempDirGuard {
        path: PathBuf,
    }

    impl TempDirGuard {
        fn new() -> Self {
            // A process-wide counter, NOT a timestamp: `SystemTime::now()`
            // has microsecond-or-worse effective granularity here (measured:
            // 19,584 of 20,000 consecutive samples were IDENTICAL), and the
            // pid is constant across every test in one binary, so "pid +
            // nanos" collides routinely between tests running in parallel.
            // Two guards then share a directory and the first `Drop` deletes
            // the other's file, which is what made
            // `config_roundtrip_preserves_engine_and_key` fail only under a
            // full workspace run.
            static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let unique = format!(
                "batchalign-dashboard-desktop-main-{}-{}",
                std::process::id(),
                COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            );
            let path = std::env::temp_dir().join(unique);
            fs::create_dir_all(&path).expect("create temp dir");
            Self { path }
        }

        fn path(&self) -> &Path {
            &self.path
        }
    }

    impl Drop for TempDirGuard {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    #[test]
    fn discover_files_recurses_filters_and_sorts() {
        let temp = TempDirGuard::new();
        let nested = temp.path().join("nested");
        fs::create_dir_all(&nested).expect("create nested dir");
        fs::write(temp.path().join("keep.CHA"), "").expect("write first file");
        fs::write(temp.path().join("skip.txt"), "").expect("write skipped file");
        fs::write(nested.join("also-keep.wav"), "").expect("write nested file");

        let files = discover_files_in_dir(temp.path(), &["cha".into(), "WAV".into()])
            .expect("discover matching files");

        assert_eq!(
            files,
            vec![
                temp.path().join("keep.CHA").to_string_lossy().into_owned(),
                nested.join("also-keep.wav").to_string_lossy().into_owned(),
            ]
        );
    }

    #[test]
    fn discover_files_with_empty_extensions_returns_all_files() {
        let temp = TempDirGuard::new();
        let nested = temp.path().join("nested");
        fs::create_dir_all(&nested).expect("create nested dir");
        fs::write(temp.path().join("alpha.cha"), "").expect("write first file");
        fs::write(temp.path().join("beta.txt"), "").expect("write second file");
        fs::write(nested.join("gamma.wav"), "").expect("write nested file");

        let files = discover_files_in_dir(temp.path(), &[]).expect("discover all files");

        assert_eq!(
            files,
            vec![
                temp.path().join("alpha.cha").to_string_lossy().into_owned(),
                temp.path().join("beta.txt").to_string_lossy().into_owned(),
                nested.join("gamma.wav").to_string_lossy().into_owned(),
            ]
        );
    }

    #[test]
    fn discover_files_rejects_non_directories() {
        let temp = TempDirGuard::new();
        let file_path = temp.path().join("file.cha");
        fs::write(&file_path, "").expect("write temp file");

        let error = discover_files_in_dir(&file_path, &[]).expect_err("reject plain file path");

        assert!(matches!(error, DiscoveryError::NotDirectory(_)));
    }
}
