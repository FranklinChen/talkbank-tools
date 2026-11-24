//! The recipe-driven command architecture.
//!
//! No longer a spike, and no longer parallel to `workflow/`, which is gone:
//! this is the load-bearing execution path, reached both by `execution/` and
//! (for now) by the legacy `runner/dispatch/` stack, which calls into it.
//! Retiring that last caller is step 4 of the phase-2 sequence.
//!
//! `catalog.rs` holds the one declaration table for released commands;
//! `recipes.rs` holds the stage recipes it points at.

pub mod catalog;
pub mod command_spec;
pub mod materialize;
pub mod planner;
pub mod recipe;
pub mod recipes;
pub mod runtime;
pub mod work_unit;
