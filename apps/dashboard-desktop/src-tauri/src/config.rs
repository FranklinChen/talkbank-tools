//! User-level configuration management for the desktop app.
//!
//! Reads and writes `~/.batchalign.ini`, the same INI file that the CLI
//! `batchalign3 setup` command manages. The desktop setup wizard uses these
//! commands instead of the terminal-based interactive flow.

use std::collections::HashMap;
use std::ffi::OsString;
use std::path::{Path, PathBuf};

use crate::protocol::{DesktopCommandAck, UserConfig};

/// Path to `~/.batchalign.ini`.
fn config_path() -> Result<PathBuf, String> {
    home_dir()
        .map(|home| home.join(".batchalign.ini"))
        .ok_or_else(|| {
            "Cannot locate your home directory; configuration was not changed".to_owned()
        })
}

/// Best-effort user home directory lookup across Unix and Windows shells.
fn home_dir() -> Option<PathBuf> {
    home_dir_from_env(|key| std::env::var_os(key))
}

/// Resolve the config path from an injected environment lookup.
#[cfg(test)]
fn config_path_from_env(mut get_var: impl FnMut(&str) -> Option<OsString>) -> Option<PathBuf> {
    home_dir_from_env(&mut get_var).map(|home| home.join(".batchalign.ini"))
}

/// Best-effort user home directory lookup across Unix and Windows shells.
fn home_dir_from_env(mut get_var: impl FnMut(&str) -> Option<OsString>) -> Option<PathBuf> {
    get_var("HOME")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .or_else(|| {
            get_var("USERPROFILE")
                .filter(|value| !value.is_empty())
                .map(PathBuf::from)
        })
        .or_else(|| {
            let drive = get_var("HOMEDRIVE")?;
            let path = get_var("HOMEPATH")?;
            Some(PathBuf::from(drive).join(path))
        })
}

fn default_user_config() -> UserConfig {
    UserConfig {
        engine: "whisper".into(),
        rev_key: None,
    }
}

#[cfg(test)]
fn is_first_launch_at(path: &Path) -> bool {
    !path.is_file()
}

fn read_config_from_path(path: &Path) -> Result<UserConfig, String> {
    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(default_user_config())
        }
        Err(error) => return Err(format!("Could not read your configuration: {error}")),
    };

    // Parse INI manually (same format as batchalign2/batchalign3 CLI).
    // Format: [asr] section with engine and engine.rev.key keys.
    let mut values: HashMap<String, String> = HashMap::new();
    let mut in_asr = false;

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') {
            in_asr = trimmed.eq_ignore_ascii_case("[asr]");
            continue;
        }
        if in_asr {
            if let Some((key, value)) = trimmed.split_once('=') {
                values.insert(key.trim().to_string(), value.trim().to_string());
            }
        }
    }

    Ok(UserConfig {
        engine: values
            .get("engine")
            .cloned()
            .unwrap_or_else(|| "whisper".into()),
        rev_key: values.get("engine.rev.key").cloned(),
    })
}

fn write_config_to_path(path: &Path, config: &UserConfig) -> Result<DesktopCommandAck, String> {
    use batchalign_types::asr_preferences::{PreferenceEdit, PreferenceEngine};
    let engine = match config.engine.as_str() {
        "rev" => PreferenceEngine::Rev,
        "whisper" => PreferenceEngine::Whisper,
        _ => return Err("Choose Rev.AI or Whisper before saving settings".to_owned()),
    };
    PreferenceEdit::new(engine, config.rev_key.clone())
        .and_then(|edit| edit.write_to(path))
        .map_err(|error| format!("Configuration was not changed: {error}"))?;
    Ok(DesktopCommandAck {
        message: format!("Config saved to {}", path.display()),
    })
}

/// True if `~/.batchalign.ini` does NOT exist, triggers the setup wizard.
#[tauri::command]
pub fn is_first_launch() -> Result<bool, String> {
    let path = config_path()?;
    match path.try_exists() {
        Ok(exists) => Ok(!exists),
        Err(error) => Err(format!("Could not check your configuration: {error}")),
    }
}

/// Read the current user config from `~/.batchalign.ini`.
///
/// Returns default fields if the file doesn't exist or is missing keys.
#[tauri::command]
pub fn read_config() -> Result<UserConfig, String> {
    read_config_from_path(&config_path()?)
}

/// Write user config to `~/.batchalign.ini`.
///
/// Creates a missing file; preserves unrelated credentials, sections and comments.
#[tauri::command]
pub fn write_config(config: UserConfig) -> Result<DesktopCommandAck, String> {
    write_config_to_path(&config_path()?, &config)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

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
                "batchalign-dashboard-desktop-config-{}-{}",
                std::process::id(),
                COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            );
            let path = std::env::temp_dir().join(unique);
            std::fs::create_dir_all(&path).expect("create temp dir");
            Self { path }
        }

        fn path(&self) -> &Path {
            &self.path
        }
    }

    impl Drop for TempDirGuard {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    #[test]
    fn config_roundtrip_preserves_engine_and_key() {
        let temp = TempDirGuard::new();
        let path = temp.path().join(".batchalign.ini");
        let config = UserConfig {
            engine: "rev".into(),
            rev_key: Some("secret-key".into()),
        };

        let message = write_config_to_path(&path, &config).expect("write config");

        assert!(message.message.contains(".batchalign.ini"));
        assert_eq!(read_config_from_path(&path).unwrap(), config);
        assert!(!is_first_launch_at(&path));
    }

    #[test]
    fn missing_config_uses_default_values() {
        let temp = TempDirGuard::new();
        let path = temp.path().join(".batchalign.ini");

        assert_eq!(read_config_from_path(&path).unwrap(), default_user_config());
        assert!(is_first_launch_at(&path));
    }

    #[test]
    fn settings_preserve_credentials_and_refuse_failed_reads_or_injection() {
        let temp = TempDirGuard::new();
        let path = temp.path().join(".batchalign.ini");
        let original = "# personal\n[asr]\nengine=rev\nengine.rev.key=retained\nengine.tencent.key=other\n[diarization]\nkey=speaker\n";
        std::fs::write(&path, original).unwrap();
        write_config_to_path(
            &path,
            &UserConfig {
                engine: "whisper".into(),
                rev_key: None,
            },
        )
        .unwrap();
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            original.replace("engine=rev", "engine = whisper")
        );
        let before = std::fs::read(&path).unwrap();
        assert!(write_config_to_path(
            &path,
            &UserConfig {
                engine: "rev\n[other]".into(),
                rev_key: None
            }
        )
        .is_err());
        assert_eq!(std::fs::read(&path).unwrap(), before);
        assert!(read_config_from_path(temp.path()).is_err());
        assert_eq!(config_path_from_env(|_| None), None);
    }

    #[test]
    fn config_path_prefers_home_then_windows_fallbacks() {
        let home_path = config_path_from_env(|key| match key {
            "HOME" => Some(OsString::from("unix-home")),
            "USERPROFILE" => Some(OsString::from("windows-home")),
            "HOMEDRIVE" => Some(OsString::from("drive")),
            "HOMEPATH" => Some(OsString::from("fallback-home")),
            _ => None,
        });
        assert_eq!(
            home_path.unwrap(),
            PathBuf::from("unix-home").join(".batchalign.ini")
        );

        let userprofile_path = config_path_from_env(|key| match key {
            "HOME" => None,
            "USERPROFILE" => Some(OsString::from("windows-home")),
            "HOMEDRIVE" => Some(OsString::from("drive")),
            "HOMEPATH" => Some(OsString::from("fallback-home")),
            _ => None,
        });
        assert_eq!(
            userprofile_path.unwrap(),
            PathBuf::from("windows-home").join(".batchalign.ini")
        );

        let drive_home_path = config_path_from_env(|key| match key {
            "HOME" => None,
            "USERPROFILE" => None,
            "HOMEDRIVE" => Some(OsString::from("drive")),
            "HOMEPATH" => Some(OsString::from("fallback-home")),
            _ => None,
        });
        assert_eq!(
            drive_home_path.unwrap(),
            PathBuf::from("drive")
                .join("fallback-home")
                .join(".batchalign.ini")
        );
    }
}
