//! Shared domain and worker-boundary types for batchalign.
//!
//! This crate is the first step toward separating build/publish boundaries:
//! worker protocol, runtime language/domain scalars, and other wire-facing
//! identifiers should not live inside the server crate.

#[macro_use]
mod macros;

pub mod api {
    //! Backward-compatible re-export of domain types historically reached via
    //! `batchalign_app::api`.
    pub use crate::domain::*;
}

pub mod domain;
pub mod paths;
pub mod scheduling;
pub mod worker;
pub mod worker_v2;
