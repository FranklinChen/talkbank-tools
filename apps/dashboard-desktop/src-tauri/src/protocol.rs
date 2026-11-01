//! Explicit transport states for desktop preferences and managed-server discovery.

use serde::{Deserialize, Serialize};

/// Only the preference fields this shell is allowed to edit.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
pub struct UserConfig {
    pub engine: String,
    pub rev_key: Option<String>,
}

/// Acknowledgement of one completed settings edit.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
pub struct DesktopCommandAck {
    pub message: String,
}

/// No PID ownership is conveyed by observing a managed server.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
#[serde(tag = "state", rename_all = "snake_case", deny_unknown_fields)]
pub enum ServerStatusInfo {
    MissingBinary,
    Stopped,
    Ready { url: String, build_hash: String },
    Refused { message: String },
}
