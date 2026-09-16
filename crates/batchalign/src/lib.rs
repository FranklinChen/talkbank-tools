#![warn(missing_docs)]
// Test code is exempt from the crate's `deny`-level panic lints: assertion
// macros panic by design, and `unwrap` on a fixture known to parse is the
// standard Rust testing idiom.
#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::unreachable,
        clippy::todo,
        clippy::unimplemented
    )
)]
//! Axum-based HTTP server for the batchalign3 processing pipeline.
//!
//! This crate implements the batchalign3 server: an axum REST API that
//! accepts NLP processing jobs (morphosyntax, forced alignment, utterance
//! segmentation, translation, coreference resolution), dispatches them to
//! Python worker processes via [`crate::worker::pool::WorkerPool`], and
//! returns results over HTTP and WebSocket.
//!
//! The server **never loads ML models directly**. All inference runs in
//! Python worker processes managed by the worker pool. The server owns the
//! CHAT format lifecycle (parse, validate, cache lookup, inject results,
//! serialize) so that Python workers are stateless `(text, lang) -> NLP`
//! inference endpoints.
//!
//! # Architecture
//!
//! ```text
//!                       +-----------+
//!                       |  CLI /    |
//!                       |  Browser  |
//!                       +-----+-----+
//!                             |
//!                    HTTP / WebSocket
//!                             |
//!                   +---------v---------+
//!                   |    axum Router    |
//!                   |  (routes, middleware)
//!                   +---------+---------+
//!                             |
//!          +------------------+------------------+
//!          |                  |                   |
//!   +------v------+   +------v------+   +--------v--------+
//!   |   JobStore  |   |   runner    |   |   WebSocket /   |
//!   | (in-memory  |   | (per-job    |   |   SSE stream    |
//!   |  + SQLite)  |   |  tokio task)|   |  (broadcast)    |
//!   +------+------+   +------+------+   +-----------------+
//!          |                  |
//!          |           +------v------+
//!          |           | WorkerPool  |
//!          |           | (semaphore  |
//!          |           |  + channel) |
//!          |           +------+------+
//!          |                  |
//!          |           stdio JSON-lines IPC
//!          |                  |
//!          |         +--------v--------+
//!          |         | Python workers  |
//!          |         | (Stanza, Whisper|
//!          |         |  Rev.AI, etc.)  |
//!          |         +-----------------+
//!          |
//!   +------v------+
//!   |   SQLite    |
//!   |  (jobs.db)  |
//!   +-------------+
//! ```
//!
//! # Endpoints
//!
//! | Method | Path                        | Description                               |
//! |--------|-----------------------------|-------------------------------------------|
//! | GET    | `/health`                   | Server version, capabilities, worker state |
//! | POST   | `/jobs/submit`              | Submit a new processing job                |
//! | GET    | `/jobs`                     | List all jobs                              |
//! | GET    | `/jobs/{id}`                | Get job details                            |
//! | GET    | `/jobs/{id}/results`        | Download completed results                 |
//! | GET    | `/jobs/{id}/results/{file}` | Download a single result file              |
//! | GET    | `/jobs/{id}/stream`         | SSE stream of real-time job progress       |
//! | DELETE | `/jobs/{id}`                | Cancel a running job                       |
//! | POST   | `/jobs/{id}/restart`        | Restart a failed/completed job             |
//! | DELETE | `/jobs/{id}/delete`         | Permanently delete a job                   |
//! | GET    | `/media/list`               | List media files from configured roots     |
//! | GET    | `/bug-reports`              | List filed bug reports                     |
//! | GET    | `/bug-reports/{id}`         | Get a single bug report                    |
//! | GET    | `/dashboard/**`             | Static dashboard SPA                       |
//! | GET    | `/ws`                       | WebSocket for real-time updates            |
//!
//! # Usage
//!
//! The primary entry point is [`create_app`], which builds the axum router
//! and shared application state. For production use, [`serve`] binds to a
//! TCP listener with graceful shutdown handling.
//!
//! ```rust,no_run
//! use batchalign::{create_app, serve};
//! use batchalign::config::ServerConfig;
//! use batchalign::worker::pool::PoolConfig;
//!
//! # async fn example() -> Result<(), batchalign::error::ServerError> {
//! // Load or construct a server config
//! let config = ServerConfig::default();
//! let pool_config = PoolConfig::default();
//!
//! // Option A: get the router for custom binding / testing
//! let (router, state) = create_app(
//!     config.clone(),
//!     pool_config.clone(),
//!     None,  // jobs_dir (default: ~/.batchalign3/jobs/)
//!     None,  // db_dir   (default: ~/.batchalign3/)
//!     None,  // build_hash
//! ).await?;
//!
//! // Option B: serve on the configured host:port with graceful shutdown
//! serve(config, pool_config, None).await?;
//! # Ok(())
//! # }
//! ```
//!
//! # Module map
//!
//! | Module          | Responsibility                                                   |
//! |-----------------|------------------------------------------------------------------|
//! | [`routes`]      | HTTP route handlers with middleware (CORS, tracing)               |
//! | [`store`]       | JobStore control plane: JobRegistry, counters, SQLite write-through, conflict detection |
//! | [`runner`]      | Per-job async tasks: dispatch to workers, track per-file progress |
//! | [`db`]          | SQLite persistence layer (WAL mode, crash recovery, TTL pruning) |
//! | [`ws`]          | WebSocket broadcast event types and channel setup                |
//! | [`media`]       | Media file resolution across configured roots with walk cache    |
//! | [`runtime_supervisor`] | Owns queue-dispatch and per-job background tasks         |
//! | [`error`]       | Typed server errors mapping to HTTP status codes                 |
//! | [`hostname`]    | Tailscale-based IP-to-hostname resolution                        |
//! | [`openapi`]     | OpenAPI 3.0 schema generation via utoipa                         |
//! | [`morphosyntax`]| Server-side morphosyntax orchestrator (parse/cache/infer/inject) |
//! | [`utseg`]       | Server-side utterance segmentation orchestrator                  |
//! | [`translate`]   | Server-side translation orchestrator                             |
//! | [`coref`]       | Server-side coreference resolution orchestrator (document-level) |
//! | [`fa`]          | Server-side forced alignment orchestrator (per-file, audio-aware)|

pub mod types;
// Re-export non-conflicting types modules at crate root for flat access.
// `types::worker` is NOT re-exported because it conflicts with `crate::worker`
// (the WorkerHandle/WorkerPool module). Access types::worker items via
// `crate::types::worker::` or the re-exports below.
pub use batchalign_types::domain::ReleasedCommand;
pub use types::{api, config, options, params, runtime, scheduling, traces};

// ── Engine modules (always available) ────────────────────────────────
pub mod benchmark;
pub mod cache;
pub(crate) mod capability;
pub mod chat_ops;
pub mod cli;
pub(crate) mod command_model;
pub mod compare;
pub mod coref;
pub mod debug_artifacts;
pub mod direct;
pub(crate) mod dispatch_language;
pub mod engine_reports;
pub mod ensure_wav;
pub mod error;
pub(crate) mod execution;
pub mod fa;
pub mod host_facts;
pub mod host_memory;
pub mod host_policy;
mod infer_retry;
pub mod media;
pub mod merge_verify;
pub(crate) mod model_manifest;
pub mod morphosyntax;
mod pipeline;
pub(crate) mod planning;
pub mod provenance;
pub(crate) mod recipe_runner;
pub(crate) mod revai;
pub mod runner;
pub mod runtime_paths;
/// What a running server publishes about itself (PID, bound port).
pub mod server_handshake;
pub mod staging;
pub mod stanza_registry;
pub mod store;
pub(crate) mod submission;
pub(crate) mod text_batch;
pub mod time;
pub mod trace_store;
pub mod transcribe;
pub mod translate;
pub mod utseg;
pub(crate) mod utseg_evidence;
pub(crate) mod utseg_route;
pub mod whisper_native;
pub mod worker;
pub mod worker_setup;
pub mod ws;

pub mod db;

// ── Server modules (require "server" feature: axum, utoipa) ───────────
#[cfg(feature = "server")]
pub mod hostname;
#[cfg(feature = "server")]
pub mod openapi;
#[cfg(feature = "server")]
pub mod routes;
#[cfg(feature = "server")]
pub(crate) mod runtime_supervisor;
#[cfg(feature = "server")]
pub mod server;
#[cfg(feature = "server")]
pub mod server_backend;
#[cfg(feature = "server")]
pub mod state;
#[cfg(feature = "server")]
pub(crate) mod websocket;

// Re-export primary API surface from submodules.
pub use direct::{DirectHost, DirectRunOutcome};

// Engine-level exports (no server deps required).
pub use worker_setup::{PreparedWorkers, RegistryDiscovery, prepare_workers};

// Server-level exports (require axum and sqlx).
#[cfg(feature = "server")]
pub use server::{
    create_app, create_app_with_prepared_workers, create_app_with_runtime, create_test_app,
    create_test_app_with_prepared_workers, serve, serve_with_runtime,
};
#[cfg(feature = "server")]
pub use state::AppState;
#[cfg(feature = "server")]
pub(crate) use websocket::ws_route;

/// This build's identity: the `git describe` string `build.rs` embeds as
/// `BUILD_HASH`. It changes on every rebuild, even when the version stays the
/// same, because staleness is judged by build identity and never by semver.
///
/// The one reader of `BUILD_HASH` in the crate. The CLI's stale-daemon check,
/// the unchecked-ASR warning, receipts and the registry's foreign-build
/// refusal all ask here, so they cannot disagree about which build this is.
pub fn build_hash() -> &'static str {
    env!("BUILD_HASH")
}

/// Create a CHAT parser handle.
///
/// The tree-sitter grammar is compiled into the binary and structurally
/// validated by CI, so construction is infallible in practice.
#[allow(clippy::expect_used)]
pub(crate) fn chat_parser() -> batchalign_transform::parse::TreeSitterParser {
    batchalign_transform::parse::TreeSitterParser::new()
        .expect("tree-sitter CHAT grammar must load")
}

/// Return whether one closed released command requires shared-filesystem audio access.
pub fn released_command_uses_local_audio(command: ReleasedCommand) -> bool {
    command_model::released_command_uses_local_audio(command)
}

/// Return whether one released command name requires shared-filesystem audio access.
///
/// This keeps the old stringly helper only for callers that still sit at a
/// trust boundary. Contributor-facing Rust code should prefer
/// [`released_command_uses_local_audio`].
pub fn command_uses_local_audio(command: &str) -> bool {
    command_model::command_uses_local_audio(command)
}

/// Return whether a command may use `paths_mode`, i.e. the CLI may
/// send filesystem paths instead of file content when submitting to a
/// local daemon. Covers audio commands (which already used paths_mode)
/// plus batched-text commands where the server-side runner reads CHAT
/// files by path.
pub fn released_command_supports_paths_mode(command: ReleasedCommand) -> bool {
    command_model::released_command_supports_paths_mode(command)
}

/// A `ChatCleanedText` built the way production builds one: by PARSING.
///
/// chatter v0.12.0 removed `ChatCleanedText::test_unchecked` and the
/// `test-utils` feature gating it, correctly: a type whose job is to prove
/// "this text came from a parsed AST" is only as strong as its weakest
/// constructor, and an escape hatch any dev-dependency could switch on was
/// that constructor. Fixtures go through the parser now, so a fixture cannot
/// assert a projection the parser would not produce.
///
/// One owner rather than a call per site, because building a parser is not
/// free and the next reader should see a single route.
#[cfg(test)]
pub(crate) fn parsed_word_text_cleaned(word: &str) -> talkbank_model::ChatCleanedText {
    let parser = talkbank_parser::TreeSitterParser::new()
        .expect("tree-sitter parser must initialise for fixtures");
    // A `match` rather than `unwrap_or_else(|e| panic!(..))`: identical
    // behaviour, but the silent-defaults ratchet matches the `unwrap_or*`
    // spelling and cannot see that this one diverges instead of substituting a
    // value. Respelling a false positive, not evading the rule; nothing here
    // invents a fixture.
    let parsed = match parser.parse_word(word) {
        Ok(parsed) => parsed,
        Err(e) => panic!("fixture {word:?} must parse as a CHAT word: {e:?}"),
    };
    talkbank_model::ChatCleanedText::from_word(&parsed)
}
