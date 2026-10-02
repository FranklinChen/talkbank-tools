//! Server configuration: mirrors `batchalign/serve/config.py`.
//!
//! Deserializes from the runtime-owned `server.yaml` under the resolved state
//! directory using yaml_serde.
//! No OmegaConf interpolation needed, plain YAML is sufficient.
//!
//! # Sub-modules
//!
//! | Module    | Purpose |
//! |-----------|---------|
//! | [`layout`]  | `RuntimeLayout`, filesystem path resolution from env/home |
//! | [`server`]  | `ServerConfig` struct, `FleetTarget`, serde defaults |
//! | [`resolve`] | `ServerConfig` methods: validation and memory-tier resolution |
//! | [`load`]    | YAML loading helpers and `ConfigError` |

mod layout;
mod lease;
mod load;
mod port;
mod positive;
mod resolve;
mod server;

mod serde_helpers;

#[cfg(test)]
mod tests;

// Re-export everything at the `config` module level for backwards compatibility.
// Callers use `crate::config::ServerConfig`, `crate::config::RuntimeLayout`, etc.
pub use layout::*;
pub use lease::{LEASE_HEARTBEAT, LeaseTtl, LeaseTtlTooShort};
pub use load::*;
pub use port::PortRequest;
pub use positive::{JobTtlDays, MemoryGatePollSeconds, WorkerStartupLimit};
pub use serde_helpers::zero_as_no_override;
pub use server::*;
