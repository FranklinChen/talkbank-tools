//! CLI and server integration tests sharing one warmed fixture process.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable,
    clippy::todo,
    clippy::unimplemented
)]

mod cli_common;
mod common;
mod live_deadline;

#[path = "cli.rs"]
mod cli;
#[path = "command_family_smoke.rs"]
mod command_family_smoke;
#[path = "command_matrix.rs"]
mod command_matrix;
#[path = "commands.rs"]
mod commands;
#[path = "daemon_e2e.rs"]
mod daemon_e2e;
#[path = "e2e.rs"]
mod e2e;
#[path = "integration.rs"]
mod integration;
#[path = "restart_handoff.rs"]
mod restart_handoff;
#[path = "serve_start_workers_persisted.rs"]
mod serve_start_workers_persisted;
#[path = "shared_test_server_smoke.rs"]
mod shared_test_server_smoke;
