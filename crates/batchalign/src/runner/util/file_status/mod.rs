//! File status tracking, progress reporting, and retry state management.
//!
//! Split into focused sub-modules:
//! - [`file_stage`]: canonical stage labels for file lifecycles
//! - [`event_sink`]: runner event sink trait and store-backed implementation
//! - [`tracker`]: per-file lifecycle helper and free state-mutation functions
//! - [`supervision`]: spawning/draining supervised file tasks and fallback cleanup
//! - [`attempts`]: one file's attempts under the retry policy, as a typestate

mod attempts;
mod event_sink;
mod file_stage;
mod supervision;
pub mod tracker;

#[cfg(test)]
pub mod test_sink;
#[cfg(test)]
mod tests;

// Re-export everything at the same paths callers already use.
// Reason: re-exports an item that is visible in this crate only; Clippy's
// suggested `pub` does not compile (E0365).
#[allow(clippy::redundant_pub_crate)]
pub(crate) use crate::store::StoreRunnerEventSink;
pub use attempts::{AfterFailure, FileAttempts};
pub use event_sink::RunnerEventSink;
pub use file_stage::FileStage;
pub use tracker::{
    FileRunTracker, FileTaskOutcome, ProgressSender, ProgressUpdate, TerminalRecorded,
};

pub use supervision::{
    SupervisedFileTasks, drain_supervised_file_tasks, observing_file_waits,
    record_file_cancelled_before_dispatch, spawn_observed_progress_forwarder,
    spawn_supervised_file_task,
};
pub use tracker::set_file_progress;
