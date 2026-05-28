//! CLI argument definitions for `talkbank` commands and global flags.
//!
//! This module is split by concern:
//! - `core` — top-level `Cli` struct and the `Commands` enum
//! - `cli_types` — shared config enums (log format, TUI mode, output format, parser backend)
//! - `cache_commands` — `chatter cache` subcommands
//! - `debug_commands` — `chatter debug` subcommands
//! - `clan_common` — shared CLAN argument groups and formats
//! - `clan_commands` — CLAN subcommands

mod cache_commands;
mod clan_commands;
mod clan_common;
mod cli_types;
mod core;
mod debug_commands;

pub use cache_commands::CacheCommands;
pub use clan_commands::{
    CapitalizationArg, ClanCommands, FreqposPositionArg, apply_clan_help_grouping,
};
pub use clan_common::{ClanOutputFormat, CommonAnalysisArgs, InheritedContextArgs};
pub use cli_types::{AlignmentTier, LogFormat, OutputFormat, ParserBackend};
pub use core::{Cli, Commands};
pub use debug_commands::DebugCommands;
