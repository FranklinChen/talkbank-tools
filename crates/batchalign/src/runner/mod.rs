//! Per-job async task: port of `JobStore._run_job` from Python.
//!
//! Each job runs as a `tokio::spawn` task. It acquires a semaphore permit,
//! then processes files concurrently (bounded by `compute_job_workers`) via
//! the WorkerPool.
//!
//! ## Sub-module layout
//!
//! - `context`: shared type definitions: execution contexts, host bundles,
//!   orchestration traits, and internal request/error enums.
//! - `execution`: job lifecycle entry points (`job_task`, `run_direct_job`,
//!   `run_server_job_attempt`) and the core `run_hosted_job` loop.
//! - `routing`: command dispatch routing: capability resolution, dispatch-
//!   family selection, and per-command dispatch wrappers.
//! - `dispatch/`: concrete dispatch family implementations (batched text,
//!   FA, transcribe, benchmark, compare, media-analysis).
//! - `policy`: command→infer-task mapping and result-filename rules.
//! - `util/`: progress tracking, event sinks, preflight helpers.
//! - `debug_dumper`: optional debug output capture.

mod context;
pub(crate) mod debug_dumper;
mod dispatch;
pub(crate) use dispatch::preserved_media_name_for_chat;
pub(crate) use dispatch::{
    AbandonedPartialAttempt, CompletedRecognition, UtrRecognitionRoute, UtrTokenConversion,
};
#[cfg(test)]
pub(crate) use dispatch::{
    PreparedRecognition, convert_utr_for_test, prepare_after_partial_failure_for_test,
};
mod execution;
mod job_scope;
pub(crate) use job_scope::current_file_span;
mod policy;
mod routing;
mod test_echo;
pub(crate) mod util;

// --- Re-exports for crate-internal consumers ---

// Types and traits from context
pub(crate) use context::{
    DirectExecutionHost, DispatchEnd, DispatchHostContext, ExecutionEngine,
    MemoryGateRejectionDisposition, QueuedJobOrchestrator, RunnerExecutionContext,
    ServerExecutionHost,
};

// The one constructor of a job-level refusal, for dispatchers outside the router.
pub(crate) use routing::refuse;

// Job lifecycle entry points from execution
pub(crate) use execution::{job_task, run_direct_job};

// Re-exports used only by tests
#[cfg(test)]
use execution::record_preflight_media_failures;
#[cfg(test)]
use policy::command_requires_chat_infer;

#[cfg(test)]
mod tests;
