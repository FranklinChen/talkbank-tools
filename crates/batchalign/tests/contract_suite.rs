//! Fast format, compatibility, CLI-surface, and workflow contracts.
//!
//! These tests share one executable so a no-op developer run does not pay a
//! separate macOS process-verification and linker cost for every source file.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable,
    clippy::todo,
    clippy::unimplemented
)]

mod cli_common;
mod live_deadline;

#[path = "chat_ops_mor_count_parity_reference_corpus.rs"]
mod chat_ops_mor_count_parity_reference_corpus;
#[path = "command_surface_manifest.rs"]
mod command_surface_manifest;
#[path = "compare_runs_cli.rs"]
mod compare_runs_cli;
#[path = "compat_contracts.rs"]
mod compat_contracts;
#[path = "json_compat.rs"]
mod json_compat;
#[path = "merge_verify.rs"]
mod merge_verify;
#[path = "utr_alignment_cli.rs"]
mod utr_alignment_cli;
#[path = "utseg_replay_cli.rs"]
mod utseg_replay_cli;
#[path = "worker_protocol_v2_compat.rs"]
mod worker_protocol_v2_compat;
#[path = "workflow_helpers.rs"]
mod workflow_helpers;
