//! Worker protocol V2 schema types shared across batchalign crates.
//!
//! Split into submodules:
//! - [`requests`] — request envelopes, task payloads, shared enums, newtypes
//! - [`responses`] — result types, execute response, progress events

pub mod requests;
pub mod responses;

pub use requests::*;
pub use responses::*;
