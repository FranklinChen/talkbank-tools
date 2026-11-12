//! Non-owning connection to the CLI-managed local server.
//!
//! Discovery belongs to the CLI, which owns handshake/config/build admission.
//! A desktop window never owns the service PID and cannot kill shared jobs.

use std::path::PathBuf;
use std::time::Duration;

use tokio::process::Command;

use crate::protocol::ServerStatusInfo;

#[derive(Debug, thiserror::Error)]
enum ConnectionError {
    #[error("Could not run the installed Batchalign CLI: {0}")]
    Launch(#[source] std::io::Error),
    #[error("The installed CLI did not finish {action} within {seconds} seconds. Check whether the operating system is waiting for file access, then retry. The managed service has not been stopped.")]
    TimedOut { action: &'static str, seconds: u64 },
    #[error("The installed CLI could not complete {action}. Update it using the Batchalign installation guide. {detail}")]
    CliRefused { action: String, detail: String },
    #[error("The installed CLI did not return a supported server observation. Update the CLI before opening the desktop app.")]
    UnsupportedObservation(#[source] serde_json::Error),
    #[error("Invalid managed server URL: {0}")]
    InvalidUrl(#[source] url::ParseError),
    #[error("The desktop requires an identified local managed server.")]
    NotLocal,
}

/// Possession means a concrete installed executable was found, not guessed.
struct InstalledCli(PathBuf);

/// Every subprocess request owns a finite wait and its child process. A stalled
/// diagnostic can neither fabricate a stopped observation nor hang startup.
struct CliRequest {
    action: CliAction,
    deadline: Duration,
}

#[derive(Clone, Copy)]
enum CliAction {
    Observe,
    Start,
}

impl CliAction {
    fn request(self) -> CliRequest {
        CliRequest {
            action: self,
            deadline: Duration::from_secs(match self {
                Self::Observe => 15,
                Self::Start => 60,
            }),
        }
    }

    fn args(self) -> &'static [&'static str] {
        match self {
            Self::Observe => &["serve", "status", "--format", "json"],
            Self::Start => &["serve", "start"],
        }
    }

    fn description(self) -> &'static str {
        match self {
            Self::Observe => "checking the local service",
            Self::Start => "starting the local service",
        }
    }
}

/// Only successful, completed requests can supply bytes to observation decoding.
struct CompletedCliRequest(Vec<u8>);

impl CliRequest {
    async fn run(self, cli: &InstalledCli) -> Result<CompletedCliRequest, ConnectionError> {
        let output = tokio::time::timeout(
            self.deadline,
            Command::new(&cli.0)
                .args(self.action.args())
                .kill_on_drop(true)
                .output(),
        )
        .await
        .map_err(|_| ConnectionError::TimedOut {
            action: self.action.description(),
            seconds: self.deadline.as_secs(),
        })?
        .map_err(ConnectionError::Launch)?;
        if !output.status.success() {
            return Err(ConnectionError::CliRefused {
                action: self.action.args().join(" "),
                detail: String::from_utf8_lossy(&output.stderr).into_owned(),
            });
        }
        Ok(CompletedCliRequest(output.stdout))
    }
}

impl InstalledCli {
    fn discover() -> Option<Self> {
        let mut dirs: Vec<PathBuf> = std::env::var_os("PATH")
            .map(|paths| std::env::split_paths(&paths).collect())
            .unwrap_or_default();
        if let Some(home) = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")) {
            dirs.push(PathBuf::from(home).join(".local/bin"));
        }
        dirs.extend([
            PathBuf::from("/opt/homebrew/bin"),
            PathBuf::from("/usr/local/bin"),
        ]);
        let name = if cfg!(windows) {
            "batchalign3.exe"
        } else {
            "batchalign3"
        };
        dirs.into_iter()
            .map(|dir| dir.join(name))
            .find(|path| path.is_file())
            .map(Self)
    }

    async fn observe(&self) -> Result<ServerStatusInfo, ConnectionError> {
        let completed = CliAction::Observe.request().run(self).await?;
        decode_observation(&completed.0)
    }
}

fn decode_observation(bytes: &[u8]) -> Result<ServerStatusInfo, ConnectionError> {
    let status: ServerStatusInfo =
        serde_json::from_slice(bytes).map_err(ConnectionError::UnsupportedObservation)?;
    // Only local path-mode inputs are supported by this shell.
    if let ServerStatusInfo::Ready {
        url, build_hash, ..
    } = &status
    {
        let parsed = url::Url::parse(url).map_err(ConnectionError::InvalidUrl)?;
        if parsed.scheme() != "http"
            || parsed.host_str() != Some("127.0.0.1")
            || parsed.port().is_none()
            || !parsed.username().is_empty()
            || parsed.password().is_some()
            || parsed.path() != "/"
            || parsed.query().is_some()
            || parsed.fragment().is_some()
            || build_hash.is_empty()
        {
            return Err(ConnectionError::NotLocal);
        }
    }
    Ok(status)
}

async fn observe() -> Result<ServerStatusInfo, ConnectionError> {
    match InstalledCli::discover() {
        Some(cli) => cli.observe().await,
        None => Ok(ServerStatusInfo::MissingBinary),
    }
}

async fn connect() -> Result<ServerStatusInfo, ConnectionError> {
    let Some(cli) = InstalledCli::discover() else {
        return Ok(ServerStatusInfo::MissingBinary);
    };
    match cli.observe().await? {
        ServerStatusInfo::Stopped => {
            // The CLI starts its persistent service. No child handle is retained.
            CliAction::Start.request().run(&cli).await?;
            cli.observe().await
        }
        observed @ (ServerStatusInfo::Ready { .. }
        | ServerStatusInfo::Refused { .. }
        | ServerStatusInfo::MissingBinary) => Ok(observed),
    }
}

/// Sign one request to this account's own local service, as the same user:
/// the installed CLI reads the key from the service's handshake, fetches a
/// fresh nonce and prints only the header value. The key never reaches this
/// process or the webview, and the proof is good for this one request.
#[tauri::command]
pub async fn sign_same_user(method: String, path: String, body: String) -> Result<String, String> {
    use tokio::io::AsyncWriteExt;
    let cli = InstalledCli::discover()
        .ok_or_else(|| "the batchalign3 CLI is not installed".to_owned())?;
    let mut child = Command::new(&cli.0)
        .args([
            "serve",
            "sign-request",
            "--method",
            &method,
            "--path",
            &path,
        ])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .map_err(|error| error.to_string())?;
    if let Some(mut stdin) = child.stdin.take() {
        stdin
            .write_all(body.as_bytes())
            .await
            .map_err(|error| error.to_string())?;
    }
    let output = tokio::time::timeout(Duration::from_secs(15), child.wait_with_output())
        .await
        .map_err(|_| "signing the request timed out".to_owned())?
        .map_err(|error| error.to_string())?;
    match output.status.success() {
        true => Ok(String::from_utf8_lossy(&output.stdout).trim().to_owned()),
        false => Err(String::from_utf8_lossy(&output.stderr).trim().to_owned()),
    }
}

/// Discover the local service without starting, stopping or replacing it.
#[tauri::command]
pub async fn server_status() -> Result<ServerStatusInfo, String> {
    observe().await.map_err(|error| error.to_string())
}

/// Reuse a compatible service or ask the installed CLI to start a missing one.
#[tauri::command]
pub async fn start_server() -> Result<ServerStatusInfo, String> {
    connect().await.map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    struct FixtureCli {
        directory: tempfile::TempDir,
        cli: InstalledCli,
    }

    #[cfg(unix)]
    impl FixtureCli {
        fn script(body: &str) -> Self {
            use std::os::unix::fs::PermissionsExt;
            let directory = tempfile::tempdir().unwrap();
            let path = directory.path().join("batchalign3");
            std::fs::write(&path, format!("#!/bin/sh\n{body}\n")).unwrap();
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o700)).unwrap();
            Self {
                directory,
                cli: InstalledCli(path),
            }
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn actual_cli_process_admits_ready_and_preserves_refusals() {
        let ready = FixtureCli::script(
            "printf '%s' '{\"state\":\"ready\",\"url\":\"http://127.0.0.1:8001\",\"build_hash\":\"fixture\"}'",
        );
        assert!(matches!(
            ready.cli.observe().await.unwrap(),
            ServerStatusInfo::Ready { .. }
        ));
        let refused = FixtureCli::script("printf '%s' 'fixture refusal' >&2; exit 7");
        assert!(
            matches!(refused.cli.observe().await, Err(ConnectionError::CliRefused { detail, .. }) if detail == "fixture refusal")
        );
        let malformed = FixtureCli::script("printf '%s' 'not an observation'");
        assert!(matches!(
            malformed.cli.observe().await,
            Err(ConnectionError::UnsupportedObservation(_))
        ));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_stalled_cli_is_reaped_without_fabricating_a_stopped_service() {
        let stalled = FixtureCli::script(
            "printf '%s' \"$$\" > \"$(dirname \"$0\")/pid\"\nexec /bin/sleep 60",
        );
        let request = CliRequest {
            action: CliAction::Observe,
            deadline: Duration::from_millis(250),
        };
        assert!(matches!(
            request.run(&stalled.cli).await,
            Err(ConnectionError::TimedOut { .. })
        ));
        let pid = std::fs::read_to_string(stalled.directory.path().join("pid")).unwrap();
        // kill_on_drop is an OS boundary, not a claim inferred from timeout.
        // Tokio may reap on its next scheduler turn, so allow a bounded wait.
        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                if !Command::new("/bin/kill")
                    .args(["-0", &pid])
                    .status()
                    .await
                    .unwrap()
                    .success()
                {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("timed-out diagnostic child must not survive");
    }

    #[test]
    fn discovery_admits_only_complete_local_observations() {
        assert!(matches!(
            decode_observation(
                br#"{"state":"ready","url":"http://127.0.0.1:8001","build_hash":"build"}"#
            )
            .unwrap(),
            ServerStatusInfo::Ready { .. }
        ));
        for invalid in [
            r#"{"state":"ready","url":"http://remote.example:8001","build_hash":"build"}"#,
            r#"{"state":"ready","url":"http://127.0.0.1:8001","build_hash":""}"#,
            r#"{"state":"ready","url":"http://127.0.0.1:8001"}"#,
            r#"{"state":"unknown"}"#,
            r#"{"running":true,"port":8001}"#,
        ] {
            assert!(decode_observation(invalid.as_bytes()).is_err(), "{invalid}");
        }
        assert!(matches!(
            decode_observation(br#"{"state":"stopped"}"#).unwrap(),
            ServerStatusInfo::Stopped
        ));
    }
}
