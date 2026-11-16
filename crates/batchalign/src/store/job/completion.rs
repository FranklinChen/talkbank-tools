//! File completion, failure, retry and progress records.
use crate::api::{ContentType, DisplayPath, FileProgressStage, MachineTime};

/// Successful result metadata for one completed file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CompletedFileOutput {
    /// Logical result filename returned to clients.
    pub filename: DisplayPath,
    /// MIME-like content type stored with the result.
    pub content_type: ContentType,
    /// What the command decided about stamping this file with provenance.
    pub stamp: crate::api::FileStampOutcome,
    /// What the producer left out of its work on purpose; recorded on the
    /// file's terminal phase, whichever it is.
    pub exclusions: Vec<crate::api::OutputExclusionRecord>,
}

/// How a file finished without failing: what the store records as its
/// terminal phase and downloadable result.
///
/// A sum rather than an `Option<CompletedFileOutput>` beside a separate
/// "diagnostics" argument: a diagnosed file always has a written result, so
/// "diagnosed without a result" and "result with diagnostics nobody reads"
/// have no spelling.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum FileCompletion {
    /// Finished with nothing to download.
    WithoutResult,
    /// Finished with a result that passed its output admission, or whose kind
    /// has none. Recorded as `FilePhase::Done`.
    Clean(CompletedFileOutput),
    /// The command's own producer generated output that failed admission, and
    /// it was written together with what the admission found. Recorded as
    /// `FilePhase::Diagnosed`; terminal, never retried, not a failure.
    Diagnosed {
        /// The written result.
        result: CompletedFileOutput,
        /// Every finding, and every stage skipped because of them.
        diagnostics: crate::api::FileOutputDiagnostics,
    },
}

impl FileCompletion {
    /// The result to record, whichever way the file finished with one.
    pub(crate) fn result(&self) -> Option<&CompletedFileOutput> {
        match self {
            Self::WithoutResult => None,
            Self::Clean(result) | Self::Diagnosed { result, .. } => Some(result),
        }
    }
}

/// Failure details for one terminal file error.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FileFailureRecord {
    /// Human-readable error message.
    pub message: String,
    /// Broad failure category for grouping and retry policy.
    pub category: crate::scheduling::FailureCategory,
    /// When the file failed, by the store's clock.
    pub finished_at: crate::store::EventTime,
}

/// Retry metadata for one transient file failure.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FileRetryRecord {
    /// Human-readable retry message shown to clients.
    pub message: String,
    /// Broad failure category for the retryable attempt.
    pub category: crate::scheduling::FailureCategory,
    /// When the failed attempt finished, by the store's clock.
    pub finished_at: crate::store::EventTime,
    /// Earliest time when the next attempt may run.
    pub retry_at: MachineTime,
}

/// Ephemeral progress update for one in-flight file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FileProgressRecord {
    /// Stable machine-readable stage code.
    pub stage: FileProgressStage,
    /// Current progress counter when available.
    pub current: Option<i64>,
    /// Total progress counter when available.
    pub total: Option<i64>,
}
