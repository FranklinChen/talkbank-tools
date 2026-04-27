//! Command implementations for the CLI.
//!
//! Each command has its own module:
//! - `validate` - File and directory validation
//! - `normalize` - CHAT normalization
//! - `json` - JSON conversion (to-json, from-json)
//! - `alignment` - Alignment visualization
//! - `watch` - Continuous validation on file changes
//! - `lint` - Auto-fixable issue detection and repair
//! - `cache` - Cache management (stats, clear)
//! - `new_file` - Create new minimal valid CHAT files
//!
//! # Related CHAT Manual Sections
//!
//! - <https://talkbank.org/0info/manuals/CHAT.html#File_Format>
//! - <https://talkbank.org/0info/manuals/CHAT.html#File_Headers>
//! - <https://talkbank.org/0info/manuals/CHAT.html#Main_Tier>
//! - <https://talkbank.org/0info/manuals/CHAT.html#Dependent_Tiers>

pub mod alignment;
pub mod cache;
pub mod clan;
pub mod clean;
pub mod debug;
mod dispatch;
pub mod find;
pub mod json;
pub mod lint;
pub mod list_checks;
pub mod new_file;
pub mod normalize;
pub mod schema;
pub mod validate;
pub mod validate_parallel;
pub mod validate_utseg;
pub mod watch;

pub use alignment::show_alignment;
pub use clan::run_clan;
pub use clean::clean_file;
pub use dispatch::{CommandContext, dispatch_command};
pub use json::{chat_to_json, json_to_chat};
pub use lint::lint_files;
pub use new_file::create_new_file;
pub use normalize::normalize_chat;
pub use schema::run_schema;
pub use validate::validate_file;
pub use validate_parallel::{
    AlignmentValidationMode, CacheRefreshMode, RoundtripValidationMode, ValidationInterface,
};
pub use watch::watch_files;
