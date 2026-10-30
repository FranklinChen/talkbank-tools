//! Worker profile verification tests, real-model resource usage assertions.
//!
//! These tests intentionally stay on the server path because they inspect
//! `/health`, live worker keys, and other server-side profile state. They verify
//! that the worker profile system correctly groups InferTasks into shared
//! workers, reducing memory consumption compared to per-task worker spawning.
//!
//! Requirements:
//! - Python 3 with batchalign installed
//! - FA models (Wave2Vec) for GPU profile tests
//! - Stanza models for Stanza profile tests
//!
//! The suite is opt-in, so a missing model fails the test rather than
//! skipping it.
//!
//! Run: `cargo test -p batchalign --features ml-golden --test ml_golden`

mod gpu;
mod helpers;
mod labels;
mod stanza;
